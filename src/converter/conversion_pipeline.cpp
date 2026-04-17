/// @file conversion_pipeline.cpp
/// @brief End-to-end model conversion pipeline implementation.

#include "converter/conversion_pipeline.h"
#include "converter/activation_collector.h"
#include "converter/kmeans.h"
#include "converter/model_config.h"
#include "converter/model_reader.h"
#include "converter/quantizer.h"
#include "converter/router_calibrator.h"
#include "format/expert_format.h"
#include "kernel/packing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>

namespace nos {

struct ConversionPipeline::Impl {
    std::unique_ptr<ModelReader> reader;
    ModelConfig model_config;
    int expert_count = 0;
    uint64_t largest_expert_bytes = 0;

    // Checkpoint state
    int completed_layers = 0;
    std::string checkpoint_path;

    bool load_checkpoint(const std::string& output_dir);
    bool save_checkpoint(const std::string& output_dir, int layer, uint64_t nxp_offset);
    void remove_checkpoint(const std::string& output_dir);
};

ConversionPipeline::ConversionPipeline() : impl_(new Impl) {}
ConversionPipeline::~ConversionPipeline() { delete impl_; }

bool ConversionPipeline::Impl::load_checkpoint(const std::string& output_dir) {
    checkpoint_path = output_dir + "/.checkpoint.json";
    std::ifstream ifs(checkpoint_path);
    if (!ifs.is_open()) return false;

    try {
        nlohmann::json j = nlohmann::json::parse(ifs);
        completed_layers = j.at("completed_layers").get<int>();
        return true;
    } catch (...) {
        return false;
    }
}

bool ConversionPipeline::Impl::save_checkpoint(const std::string& output_dir,
                                                 int layer, uint64_t nxp_offset) {
    std::string tmp_path = output_dir + "/.checkpoint.json.tmp";
    checkpoint_path = output_dir + "/.checkpoint.json";

    nlohmann::json j = {
        {"completed_layers", layer},
        {"nxp_offset", nxp_offset}
    };

    // Write to temp, then rename atomically
    {
        std::ofstream ofs(tmp_path);
        if (!ofs.is_open()) return false;
        ofs << j.dump(2);
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::rename(tmp_path, checkpoint_path, ec);
    return !ec;
}

void ConversionPipeline::Impl::remove_checkpoint(const std::string& output_dir) {
    std::string path = output_dir + "/.checkpoint.json";
    std::filesystem::remove(path);
}

bool ConversionPipeline::run(const ConversionConfig& config) {
    namespace fs = std::filesystem;

    // Create output directory
    fs::create_directories(config.output_dir);

    // --- Open input model ---
    impl_->reader = ModelReader::create(config.input_path);
    if (!impl_->reader) {
        std::fprintf(stderr, "ERROR: Failed to open model at '%s'\n",
                    config.input_path.c_str());
        return false;
    }

    impl_->model_config = impl_->reader->config();
    auto& mc = impl_->model_config;

    if (mc.n_layers == 0 || mc.hidden_dim == 0 || mc.intermediate_dim == 0) {
        std::fprintf(stderr, "ERROR: Invalid model config (layers=%u, hidden=%u, inter=%u)\n",
                    mc.n_layers, mc.hidden_dim, mc.intermediate_dim);
        return false;
    }

    // --- Compute expert count per layer ---
    // FFN weight bytes: gate_proj + up_proj + down_proj, each [intermediate_dim x hidden_dim] as FP16
    uint64_t ffn_weight_bytes = static_cast<uint64_t>(mc.intermediate_dim)
                              * static_cast<uint64_t>(mc.hidden_dim) * 2 * 3;
    uint64_t target_bytes = static_cast<uint64_t>(config.target_expert_size_mb) * 1024 * 1024;
    impl_->expert_count = std::max(2, static_cast<int>(
        std::ceil(static_cast<double>(ffn_weight_bytes) / static_cast<double>(target_bytes))));

    std::fprintf(stderr, "INFO: Model: %s, %u layers, hidden=%u, inter=%u\n",
                mc.architecture.c_str(), mc.n_layers, mc.hidden_dim, mc.intermediate_dim);
    std::fprintf(stderr, "INFO: Expert count per layer: %d (target %d MB/expert)\n",
                impl_->expert_count, config.target_expert_size_mb);

    // --- Load checkpoint if resuming ---
    int start_layer = 0;
    if (config.resume) {
        if (impl_->load_checkpoint(config.output_dir)) {
            start_layer = impl_->completed_layers;
            std::fprintf(stderr, "INFO: Resuming from layer %d\n", start_layer);
        }
    }

    // --- Stage 1: Collect activations ---
    std::fprintf(stderr, "INFO: Stage 1/5: Collecting activations...\n");
    ActivationCollectorConfig act_config;
    act_config.calibration_data_path = config.calibration_data_path;
    act_config.calibration_samples = config.calibration_samples;
    auto activations = collect_activations(*impl_->reader, act_config);

    if (activations.per_layer_magnitudes.size() != mc.n_layers) {
        std::fprintf(stderr, "ERROR: Activation collection returned %zu layers, expected %u\n",
                    activations.per_layer_magnitudes.size(), mc.n_layers);
        return false;
    }

    // --- Open NXP writer ---
    NxpFileHeader nxp_header{};
    nxp_header.magic = NXP_MAGIC;
    nxp_header.version = NXP_VERSION;
    nxp_header.num_layers = mc.n_layers;
    nxp_header.experts_per_layer = static_cast<uint32_t>(impl_->expert_count);
    nxp_header.hidden_dim = mc.hidden_dim;
    nxp_header.intermediate_dim = mc.intermediate_dim;
    nxp_header.packing_mode = 0;  // 5-per-byte
    nxp_header.scale_dtype = 0;   // FP16
    nxp_header.total_experts = static_cast<uint64_t>(mc.n_layers)
                             * static_cast<uint64_t>(impl_->expert_count);
    std::memset(nxp_header.reserved, 0, sizeof(nxp_header.reserved));

    std::string nxp_path = config.output_dir + "/model.nxp";
    NxpWriter writer;

    // For resume: if nxp file exists and we're resuming, we need to rewrite
    // (NxpWriter always starts fresh -- for production, we'd append)
    if (start_layer > 0 && !fs::exists(nxp_path)) {
        // Checkpoint exists but nxp file doesn't -- restart from scratch
        start_layer = 0;
    }

    if (!writer.open(nxp_path, nxp_header)) {
        std::fprintf(stderr, "ERROR: Failed to open NXP file '%s'\n", nxp_path.c_str());
        return false;
    }

    // --- Stage 2: Per-layer conversion ---
    std::fprintf(stderr, "INFO: Stage 2/5: Converting layers...\n");

    int n_layers = static_cast<int>(mc.n_layers);
    int hidden_dim = static_cast<int>(mc.hidden_dim);
    int intermediate_dim = static_cast<int>(mc.intermediate_dim);
    int expert_count = impl_->expert_count;

    // Calibration embeddings for router recalibration (reuse from activation collection)
    // We'll generate simple synthetic inputs for router calibration
    int n_cal_samples = std::min(config.calibration_samples, 64);
    std::vector<float> cal_inputs(static_cast<size_t>(n_cal_samples)
                                * static_cast<size_t>(hidden_dim), 0.01f);
    // Simple deterministic pattern
    for (int s = 0; s < n_cal_samples; s++) {
        for (int d = 0; d < hidden_dim; d++) {
            cal_inputs[static_cast<size_t>(s) * static_cast<size_t>(hidden_dim)
                     + static_cast<size_t>(d)] =
                std::sin(static_cast<float>(s * hidden_dim + d) * 0.01f) * 0.1f;
        }
    }

    // Store all router weights for stage 3
    std::vector<RouterWeights> all_routers;
    all_routers.resize(static_cast<size_t>(n_layers));

    for (int layer = 0; layer < n_layers; layer++) {
        std::fprintf(stderr, "  Layer %d/%d", layer + 1, n_layers);

        if (layer < start_layer) {
            std::fprintf(stderr, " (skipped - checkpoint)\n");
            continue;
        }

        // 2a. Load FFN weights (gate_proj, up_proj, down_proj)
        std::vector<std::string> gate_names = {
            "model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight",
            "blk." + std::to_string(layer) + ".ffn_gate.weight"
        };
        std::vector<std::string> up_names = {
            "model.layers." + std::to_string(layer) + ".mlp.up_proj.weight",
            "blk." + std::to_string(layer) + ".ffn_up.weight"
        };
        std::vector<std::string> down_names = {
            "model.layers." + std::to_string(layer) + ".mlp.down_proj.weight",
            "blk." + std::to_string(layer) + ".ffn_down.weight"
        };

        auto find_tensor = [&](const std::vector<std::string>& names) -> const TensorInfo* {
            for (const auto& n : names) {
                auto* info = impl_->reader->find_tensor(n);
                if (info) return info;
            }
            return nullptr;
        };

        const TensorInfo* gate_info = find_tensor(gate_names);
        const TensorInfo* up_info = find_tensor(up_names);
        const TensorInfo* down_info = find_tensor(down_names);

        bool has_ffn = gate_info && up_info && down_info;

        if (has_ffn) {
            // Load gate_proj as FP16 [intermediate_dim x hidden_dim]
            size_t ffn_elems = static_cast<size_t>(intermediate_dim)
                             * static_cast<size_t>(hidden_dim);
            std::vector<uint16_t> gate_fp16(ffn_elems);
            std::vector<uint16_t> up_fp16(ffn_elems);
            // down_proj: [hidden_dim x intermediate_dim]
            std::vector<uint16_t> down_fp16(ffn_elems);

            auto load_fp16 = [&](const TensorInfo* info, std::vector<uint16_t>& buf) {
                if (info->dtype == "F16") {
                    impl_->reader->read_tensor(*info, buf.data(),
                                              buf.size() * sizeof(uint16_t));
                } else if (info->dtype == "F32") {
                    std::vector<float> f32(buf.size());
                    impl_->reader->read_tensor(*info, f32.data(),
                                              f32.size() * sizeof(float));
                    for (size_t i = 0; i < f32.size(); i++) {
                        buf[i] = fp32_to_fp16(f32[i]);
                    }
                }
            };

            load_fp16(gate_info, gate_fp16);
            load_fp16(up_info, up_fp16);
            load_fp16(down_info, down_fp16);

            // 2b. Cluster neurons based on activation magnitudes
            const auto& magnitudes = activations.per_layer_magnitudes[static_cast<size_t>(layer)];

            // Create per-neuron feature vectors for clustering
            // Each neuron's feature = its activation magnitude (1-dim for simplicity)
            std::vector<float> cluster_data(static_cast<size_t>(intermediate_dim));
            for (int n = 0; n < intermediate_dim; n++) {
                cluster_data[static_cast<size_t>(n)] =
                    (static_cast<size_t>(n) < magnitudes.size())
                    ? magnitudes[static_cast<size_t>(n)] : 0.0f;
            }

            KMeansResult clusters = balanced_kmeans(
                cluster_data.data(), intermediate_dim, 1, expert_count, 50,
                static_cast<uint64_t>(layer * 37 + 42));

            // 2c. Slice and quantize each expert
            std::vector<QuantizedWeights> layer_experts;
            layer_experts.reserve(static_cast<size_t>(expert_count));

            for (int e = 0; e < expert_count; e++) {
                const auto& indices = clusters.clusters[static_cast<size_t>(e)];
                int expert_rows = static_cast<int>(indices.size());
                if (expert_rows == 0) continue;

                // Slice gate_proj rows for this expert's neurons
                // gate_proj shape: [intermediate_dim x hidden_dim]
                // We take the rows corresponding to cluster's neuron indices
                std::vector<uint16_t> expert_gate(static_cast<size_t>(expert_rows)
                                                * static_cast<size_t>(hidden_dim));
                std::vector<uint16_t> expert_up(static_cast<size_t>(expert_rows)
                                              * static_cast<size_t>(hidden_dim));
                // down_proj: [hidden_dim x intermediate_dim] -- slice columns
                // For simplicity in packing, we transpose the slice to [expert_rows x hidden_dim]

                // down_proj: [hidden_dim x intermediate_dim] -- slice columns
                std::vector<uint16_t> expert_down(static_cast<size_t>(hidden_dim)
                                                 * static_cast<size_t>(expert_rows));

                for (int r = 0; r < expert_rows; r++) {
                    uint32_t neuron_idx = indices[static_cast<size_t>(r)];
                    size_t src_offset = static_cast<size_t>(neuron_idx)
                                      * static_cast<size_t>(hidden_dim);
                    size_t dst_offset = static_cast<size_t>(r)
                                      * static_cast<size_t>(hidden_dim);
                    // gate_proj row
                    std::memcpy(&expert_gate[dst_offset], &gate_fp16[src_offset],
                               static_cast<size_t>(hidden_dim) * sizeof(uint16_t));
                    // up_proj row
                    std::memcpy(&expert_up[dst_offset], &up_fp16[src_offset],
                               static_cast<size_t>(hidden_dim) * sizeof(uint16_t));
                }

                // Slice down_proj columns: down_fp16[row * inter_dim + neuron_idx]
                for (int r = 0; r < hidden_dim; r++) {
                    for (int c = 0; c < expert_rows; c++) {
                        uint32_t neuron_idx = indices[static_cast<size_t>(c)];
                        expert_down[static_cast<size_t>(r) * static_cast<size_t>(expert_rows)
                                   + static_cast<size_t>(c)] =
                            down_fp16[static_cast<size_t>(r) * static_cast<size_t>(intermediate_dim)
                                     + static_cast<size_t>(neuron_idx)];
                    }
                }

                // 2d. Ternary quantize gate+up: [2*expert_rows x hidden_dim]
                int gu_rows = expert_rows * 2;
                std::vector<uint16_t> gu_combined(static_cast<size_t>(gu_rows)
                                                 * static_cast<size_t>(hidden_dim));
                std::memcpy(gu_combined.data(), expert_gate.data(),
                           expert_gate.size() * sizeof(uint16_t));
                std::memcpy(gu_combined.data() + expert_gate.size(), expert_up.data(),
                           expert_up.size() * sizeof(uint16_t));

                auto gu_quant = ternary_quantize(gu_combined.data(), gu_rows, hidden_dim);

                // Ternary quantize down_proj: [hidden_dim x expert_rows]
                auto down_quant = ternary_quantize(expert_down.data(), hidden_dim, expert_rows);

                // Concatenate packed data: [gate+up packed] [down packed]
                std::vector<uint8_t> all_packed;
                all_packed.reserve(gu_quant.packed.size() + down_quant.packed.size());
                all_packed.insert(all_packed.end(), gu_quant.packed.begin(), gu_quant.packed.end());
                all_packed.insert(all_packed.end(), down_quant.packed.begin(), down_quant.packed.end());

                // Concatenate scales: [gate+up scales] [down scales]
                std::vector<uint16_t> all_scales;
                all_scales.reserve(gu_quant.scales.size() + down_quant.scales.size());
                all_scales.insert(all_scales.end(), gu_quant.scales.begin(), gu_quant.scales.end());
                all_scales.insert(all_scales.end(), down_quant.scales.begin(), down_quant.scales.end());

                uint32_t total_channels = static_cast<uint32_t>(gu_quant.rows + down_quant.rows);
                layer_experts.push_back(gu_quant);

                // 2f. Write expert to .nxp (gate+up+down packed data, combined scales)
                auto entry = writer.write_expert(
                    static_cast<uint32_t>(layer),
                    static_cast<uint32_t>(e),
                    all_packed.data(), all_packed.size(),
                    all_scales.data(), total_channels);

                if (entry.size > impl_->largest_expert_bytes) {
                    impl_->largest_expert_bytes = entry.size;
                }
            }

            // 2e. Initialize router weights from clustering centroids
            all_routers[static_cast<size_t>(layer)] = init_from_centroids(
                clusters, cluster_data.data(), intermediate_dim, hidden_dim);

            // 2g. Quantize attention weights (Q, K, V, O) to INT8
            std::vector<std::string> attn_names = {
                "model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight",
                "model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight",
                "model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight",
                "model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight",
                "blk." + std::to_string(layer) + ".attn_q.weight",
                "blk." + std::to_string(layer) + ".attn_k.weight",
                "blk." + std::to_string(layer) + ".attn_v.weight",
                "blk." + std::to_string(layer) + ".attn_output.weight"
            };

            uint32_t attn_expert_id = 0xFFFFFFF0u;
            for (const auto& aname : attn_names) {
                const TensorInfo* ainfo = impl_->reader->find_tensor(aname);
                if (!ainfo) continue;

                int arows = static_cast<int>(ainfo->shape[0]);
                int acols = (ainfo->shape.size() > 1) ? static_cast<int>(ainfo->shape[1]) : 1;

                std::vector<uint16_t> attn_fp16(static_cast<size_t>(arows)
                                              * static_cast<size_t>(acols));
                if (ainfo->dtype == "F16") {
                    impl_->reader->read_tensor(*ainfo, attn_fp16.data(),
                                              attn_fp16.size() * sizeof(uint16_t));
                } else if (ainfo->dtype == "F32") {
                    std::vector<float> f32(attn_fp16.size());
                    impl_->reader->read_tensor(*ainfo, f32.data(),
                                              f32.size() * sizeof(float));
                    for (size_t i = 0; i < f32.size(); i++) {
                        attn_fp16[i] = fp32_to_fp16(f32[i]);
                    }
                }

                auto int8_weights = int8_quantize(attn_fp16.data(), arows, acols);

                // Store INT8 as raw bytes in .nxp with reserved expert_id
                writer.write_expert(
                    static_cast<uint32_t>(layer),
                    attn_expert_id++,
                    reinterpret_cast<const uint8_t*>(int8_weights.data.data()),
                    int8_weights.data.size(),
                    int8_weights.scales.data(),
                    static_cast<uint32_t>(int8_weights.rows));
            }
        }

        // 2h. Save per-layer checkpoint
        impl_->save_checkpoint(config.output_dir, layer + 1, 0);
        std::fprintf(stderr, " [done]\n");
    }

    // --- Stage 3: Router re-calibration ---
    std::fprintf(stderr, "INFO: Stage 3/5: Router re-calibration...\n");
    for (int layer = 0; layer < n_layers; layer++) {
        if (all_routers[static_cast<size_t>(layer)].num_experts == 0) continue;

        // We pass empty quantized_experts since the actual recalibration
        // uses the router weights + calibration inputs (gradient-free)
        std::vector<QuantizedWeights> dummy_experts;
        recalibrate_router(all_routers[static_cast<size_t>(layer)], dummy_experts,
                          cal_inputs.data(), n_cal_samples);

        // Write router weights to .nxp as a special entry
        auto& router = all_routers[static_cast<size_t>(layer)];
        std::vector<uint16_t> router_fp16(router.weights.size());
        for (size_t i = 0; i < router.weights.size(); i++) {
            router_fp16[i] = fp32_to_fp16(router.weights[i]);
        }

        // Use reserved expert_id for router weights
        writer.write_expert(
            static_cast<uint32_t>(layer),
            0xFFFFFFFFu,  // router weights sentinel
            reinterpret_cast<const uint8_t*>(router_fp16.data()),
            router_fp16.size() * sizeof(uint16_t),
            nullptr, 0);
    }

    // --- Stage 4: Write non-expert tensors ---
    std::fprintf(stderr, "INFO: Stage 4/5: Writing embeddings and output projection...\n");

    // Helper: read a tensor as FP16 bytes, converting from FP32 if needed
    auto read_tensor_as_fp16 = [&](const TensorInfo* info,
                                   std::vector<uint8_t>& out_buf) -> bool {
        if (info->dtype == "F32") {
            size_t data_size = static_cast<size_t>(info->end - info->begin);
            size_t num_elems = data_size / sizeof(float);
            std::vector<float> f32(num_elems);
            if (!impl_->reader->read_tensor(*info, f32.data(), data_size))
                return false;
            std::vector<uint16_t> fp16(num_elems);
            for (size_t i = 0; i < num_elems; i++)
                fp16[i] = fp32_to_fp16(f32[i]);
            out_buf.resize(num_elems * sizeof(uint16_t));
            std::memcpy(out_buf.data(), fp16.data(), out_buf.size());
        } else {
            size_t data_size = static_cast<size_t>(info->end - info->begin);
            out_buf.resize(data_size);
            if (!impl_->reader->read_tensor(*info, out_buf.data(), data_size))
                return false;
        }
        return true;
    };

    // Embedding matrix
    std::vector<std::string> emb_names = {
        "model.embed_tokens.weight", "token_embd.weight", "tok_embeddings.weight"
    };
    for (const auto& name : emb_names) {
        const TensorInfo* info = impl_->reader->find_tensor(name);
        if (!info) continue;

        std::vector<uint8_t> buf;
        if (read_tensor_as_fp16(info, buf)) {
            writer.write_expert(
                UINT32_MAX,  // reserved layer_id for non-layer tensors
                0,           // embedding tensor
                buf.data(), buf.size(),
                nullptr, 0);
        }
        break;
    }

    // Output projection (fall back to embedding for tied-weight models)
    std::vector<std::string> lm_head_names = {
        "lm_head.weight", "output.weight"
    };
    bool wrote_output_proj = false;
    for (const auto& name : lm_head_names) {
        const TensorInfo* info = impl_->reader->find_tensor(name);
        if (!info) continue;

        std::vector<uint8_t> buf;
        if (read_tensor_as_fp16(info, buf)) {
            writer.write_expert(
                UINT32_MAX,
                1,  // output projection
                buf.data(), buf.size(),
                nullptr, 0);
            wrote_output_proj = true;
        }
        break;
    }

    // Tied embeddings: if no lm_head found, reuse embed_tokens as output projection
    if (!wrote_output_proj) {
        std::fprintf(stderr, "INFO: No lm_head found, using tied embeddings for output projection\n");
        for (const auto& name : emb_names) {
            const TensorInfo* info = impl_->reader->find_tensor(name);
            if (!info) continue;

            std::vector<uint8_t> buf;
            if (read_tensor_as_fp16(info, buf)) {
                writer.write_expert(
                    UINT32_MAX,
                    1,  // output projection (tied to embedding)
                    buf.data(), buf.size(),
                    nullptr, 0);
            }
            break;
        }
    }

    // --- Stage 4b: Write RMSNorm weights ---
    std::fprintf(stderr, "INFO: Stage 4b/5: Writing RMSNorm weights...\n");
    {
        // Collect all norm weights: [attn_norm_L0, ffn_norm_L0, ..., final_norm]
        // Total: (n_layers * 2 + 1) * hidden_dim floats stored as FP16
        size_t total_norm_elems = (static_cast<size_t>(n_layers) * 2 + 1)
                                 * static_cast<size_t>(hidden_dim);
        std::vector<uint16_t> all_norms(total_norm_elems);
        // Initialize to 1.0 (identity) in case some norms are missing
        uint16_t one_fp16 = fp32_to_fp16(1.0f);
        std::fill(all_norms.begin(), all_norms.end(), one_fp16);

        for (int layer = 0; layer < n_layers; layer++) {
            // Attention norm (input_layernorm)
            std::vector<std::string> attn_norm_names = {
                "model.layers." + std::to_string(layer) + ".input_layernorm.weight",
                "blk." + std::to_string(layer) + ".attn_norm.weight"
            };
            for (const auto& name : attn_norm_names) {
                const TensorInfo* info = impl_->reader->find_tensor(name);
                if (!info) continue;
                size_t offset = (static_cast<size_t>(layer) * 2)
                              * static_cast<size_t>(hidden_dim);
                if (info->dtype == "F16") {
                    impl_->reader->read_tensor(*info, &all_norms[offset],
                                              static_cast<size_t>(hidden_dim) * sizeof(uint16_t));
                } else if (info->dtype == "F32") {
                    std::vector<float> f32(static_cast<size_t>(hidden_dim));
                    impl_->reader->read_tensor(*info, f32.data(),
                                              f32.size() * sizeof(float));
                    for (int d = 0; d < hidden_dim; d++) {
                        all_norms[offset + static_cast<size_t>(d)] = fp32_to_fp16(f32[static_cast<size_t>(d)]);
                    }
                }
                break;
            }

            // FFN norm (post_attention_layernorm)
            std::vector<std::string> ffn_norm_names = {
                "model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight",
                "blk." + std::to_string(layer) + ".ffn_norm.weight"
            };
            for (const auto& name : ffn_norm_names) {
                const TensorInfo* info = impl_->reader->find_tensor(name);
                if (!info) continue;
                size_t offset = (static_cast<size_t>(layer) * 2 + 1)
                              * static_cast<size_t>(hidden_dim);
                if (info->dtype == "F16") {
                    impl_->reader->read_tensor(*info, &all_norms[offset],
                                              static_cast<size_t>(hidden_dim) * sizeof(uint16_t));
                } else if (info->dtype == "F32") {
                    std::vector<float> f32(static_cast<size_t>(hidden_dim));
                    impl_->reader->read_tensor(*info, f32.data(),
                                              f32.size() * sizeof(float));
                    for (int d = 0; d < hidden_dim; d++) {
                        all_norms[offset + static_cast<size_t>(d)] = fp32_to_fp16(f32[static_cast<size_t>(d)]);
                    }
                }
                break;
            }
        }

        // Final norm
        std::vector<std::string> final_norm_names = {
            "model.norm.weight", "output_norm.weight"
        };
        for (const auto& name : final_norm_names) {
            const TensorInfo* info = impl_->reader->find_tensor(name);
            if (!info) continue;
            size_t offset = static_cast<size_t>(n_layers) * 2
                          * static_cast<size_t>(hidden_dim);
            if (info->dtype == "F16") {
                impl_->reader->read_tensor(*info, &all_norms[offset],
                                          static_cast<size_t>(hidden_dim) * sizeof(uint16_t));
            } else if (info->dtype == "F32") {
                std::vector<float> f32(static_cast<size_t>(hidden_dim));
                impl_->reader->read_tensor(*info, f32.data(),
                                          f32.size() * sizeof(float));
                for (int d = 0; d < hidden_dim; d++) {
                    all_norms[offset + static_cast<size_t>(d)] = fp32_to_fp16(f32[static_cast<size_t>(d)]);
                }
            }
            break;
        }

        writer.write_expert(
            UINT32_MAX,
            2,  // RMSNorm weights
            reinterpret_cast<const uint8_t*>(all_norms.data()),
            all_norms.size() * sizeof(uint16_t),
            nullptr, 0);
    }

    // --- Stage 5: Finalize ---
    std::fprintf(stderr, "INFO: Stage 5/5: Finalizing...\n");

    if (!writer.finalize()) {
        std::fprintf(stderr, "ERROR: Failed to finalize NXP file\n");
        return false;
    }

    // Write model_config.json
    ModelConfig out_config = mc;
    out_config.expert_count = static_cast<uint32_t>(impl_->expert_count);
    out_config.top_k = static_cast<uint32_t>(config.top_k);
    out_config.max_expert_size = static_cast<uint32_t>(impl_->largest_expert_bytes);
    out_config.attention_type = derive_attention_type(out_config.n_heads, out_config.n_kv_heads);

    nlohmann::json j = out_config;
    std::string config_path = config.output_dir + "/model_config.json";
    {
        std::ofstream ofs(config_path);
        if (!ofs.is_open()) {
            std::fprintf(stderr, "ERROR: Failed to write model_config.json\n");
            return false;
        }
        ofs << j.dump(2);
    }

    // Remove checkpoint on successful completion
    impl_->remove_checkpoint(config.output_dir);

    std::fprintf(stderr, "INFO: Conversion complete. Output: %s\n",
                config.output_dir.c_str());
    return true;
}

}  // namespace nos
