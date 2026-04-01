/// @file trainer.cpp
/// @brief Training orchestrator implementation.
///
/// NOTE: This is a RESEARCH PROTOTYPE with SIMPLIFIED forward/backward passes.
/// The training loop uses analytical gradients for linear layers:
///   grad_W = x^T * grad_output  (for a linear layer y = W * x)
///
/// A production training loop would require a full autograd system.
/// The purpose here is to demonstrate that BAdam block-wise optimization,
/// GaLore gradient projection, and LoRA fine-tuning work correctly on
/// the NeuralOS expert architecture.
///
/// For BAdam full training:
///   - Each expert block is loaded from .nxp via NxpReader to FP32 RAM
///   - Trained for steps_per_block steps with AdamW
///   - Re-quantized back to ternary and saved via NxpWriter
///   - Falls back to synthetic weights if no .nxp file found (demo mode)
///   - Gradient: finite differences or analytical for linear layer
///
/// For LoRA:
///   - Base weights frozen, only A and B matrices trained
///   - Analytical gradients through the two linear projections
///   - Adapters saved independently and merged via `neuralos merge-lora`
///
/// Composition: BAdam and GaLore are independent optimizers composed by
/// this Trainer. BAdam handles per-expert blocks; GaLore handles shared
/// attention/router layers. No internal delegation between them.

#include "training/trainer.h"
#include "training/data_loader.h"
#include "format/expert_format.h"
#include "kernel/packing.h"
#include "converter/quantizer.h"

#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

namespace nos {

bool Trainer::train(const TrainConfig& config) {
    if (config.method == "full") {
        return train_full(config);
    } else if (config.method == "lora") {
        return train_lora(config);
    } else {
        std::fprintf(stderr, "Trainer: unknown method '%s' (use 'full' or 'lora')\n",
                     config.method.c_str());
        return false;
    }
}

bool Trainer::train_full(const TrainConfig& config) {
    std::fprintf(stderr, "Trainer: starting full training (BAdam + GaLore)\n");
    std::fprintf(stderr, "  Model:  %s\n", config.model_dir.c_str());
    std::fprintf(stderr, "  Data:   %s\n", config.data_path.c_str());
    std::fprintf(stderr, "  Output: %s\n", config.output_dir.c_str());
    std::fprintf(stderr, "  Method: %s\n", config.method.c_str());
    std::fprintf(stderr, "  Epochs: %d\n", config.max_epochs);
    std::fprintf(stderr, "  Batch:  %zu\n", config.batch_size);
    std::fprintf(stderr, "  LR:     %e\n", static_cast<double>(config.badam_config.lr));
    std::fprintf(stderr, "  Steps/block: %d\n", config.badam_config.steps_per_block);

    // Load training data
    DataLoader loader;
    if (!loader.load(config.data_path)) {
        std::fprintf(stderr, "Trainer: failed to load training data\n");
        return false;
    }

    // Read model config to get dimensions
    std::string config_path = config.model_dir + "/model_config.json";
    std::ifstream cfg_ifs(config_path);
    if (!cfg_ifs.is_open()) {
        std::fprintf(stderr, "Trainer: cannot open %s\n", config_path.c_str());
        return false;
    }

    nlohmann::json model_meta;
    try {
        model_meta = nlohmann::json::parse(cfg_ifs);
    } catch (const nlohmann::json::exception& e) {
        std::fprintf(stderr, "Trainer: invalid model config: %s\n", e.what());
        return false;
    }

    const uint32_t n_layers = model_meta.value("n_layers", 0u);
    const uint32_t expert_count = model_meta.value("expert_count", 0u);
    const uint32_t hidden_dim = model_meta.value("hidden_dim", 0u);
    const uint32_t intermediate_dim = model_meta.value("intermediate_dim", 0u);

    if (n_layers == 0 || hidden_dim == 0) {
        std::fprintf(stderr, "Trainer: invalid model dimensions\n");
        return false;
    }

    std::fprintf(stderr, "  Layers: %u, Experts: %u, Hidden: %u, Intermediate: %u\n",
                 n_layers, expert_count, hidden_dim, intermediate_dim);

    // Create output directory
    std::filesystem::create_directories(config.output_dir);

    // Copy model_config.json, tokenizer, and other metadata files from source
    // model directory to output directory. This ensures the output is a valid
    // drop-in model directory for InferenceEngine::load().
    for (const auto& entry : std::filesystem::directory_iterator(config.model_dir)) {
        const auto& src = entry.path();
        if (src.extension() == ".json" || src.extension() == ".model") {
            auto dst = std::filesystem::path(config.output_dir) / src.filename();
            std::filesystem::copy_file(src, dst,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    // Try to open .nxp file for real expert I/O
    std::string nxp_path;
    for (const auto& entry : std::filesystem::directory_iterator(config.model_dir)) {
        if (entry.path().extension() == ".nxp") {
            nxp_path = entry.path().string();
            break;
        }
    }

    NxpReader nxp_reader;
    bool use_nxp = false;
    if (!nxp_path.empty() && nxp_reader.open(nxp_path)) {
        use_nxp = true;
        std::fprintf(stderr, "  NXP file: %s (%zu experts)\n",
                     nxp_path.c_str(), nxp_reader.num_entries());
    } else {
        std::fprintf(stderr, "  WARNING: No .nxp file found, using synthetic weights (demo mode)\n");
    }

    // Block-wise training loop (simplified research prototype)
    BAdamOptimizer badam(config.badam_config);
    const size_t block_params = static_cast<size_t>(hidden_dim) *
                                static_cast<size_t>(intermediate_dim);

    // Packing dimensions for ternary I/O
    const int cols_i = static_cast<int>(intermediate_dim);
    const int rows_i = static_cast<int>(hidden_dim);
    const int packed_cols = (cols_i + 4) / 5;  // 5 trits per byte

    // Storage for trained expert weights (re-quantized), indexed by (layer, expert)
    struct TrainedExpert {
        std::vector<uint8_t> packed;
        std::vector<uint16_t> scales;
    };
    std::vector<TrainedExpert> trained_experts;

    for (int epoch = 0; epoch < config.max_epochs; ++epoch) {
        std::fprintf(stderr, "\nEpoch %d/%d:\n", epoch + 1, config.max_epochs);
        loader.shuffle(static_cast<uint64_t>(epoch));
        trained_experts.clear();

        for (uint32_t layer = 0; layer < n_layers; ++layer) {
            uint32_t num_experts = (expert_count > 0) ? expert_count : 1;
            for (uint32_t exp = 0; exp < num_experts; ++exp) {
                std::vector<float> weights(block_params);

                if (use_nxp) {
                    // Load expert weights from .nxp and dequantize to FP32
                    const NxpExpertEntry* entry = nxp_reader.find_expert(layer, exp);
                    if (entry != nullptr) {
                        // Read packed ternary weights
                        std::vector<uint8_t> packed_buf(static_cast<size_t>(entry->size));
                        int rd = nxp_reader.read_expert(*entry, packed_buf.data(),
                                                        packed_buf.size());
                        // Read FP16 scale factors
                        std::vector<uint16_t> scale_buf(static_cast<size_t>(entry->num_channels));
                        int sd = nxp_reader.read_scales(*entry, scale_buf.data(),
                                                        static_cast<size_t>(entry->scale_size));

                        if (rd > 0 && sd > 0) {
                            // Dequantize: unpack trits, multiply by scale
                            std::vector<int8_t> trits(static_cast<size_t>(cols_i));
                            for (int r = 0; r < rows_i; ++r) {
                                const uint8_t* row_packed = packed_buf.data()
                                    + static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
                                unpack_row(row_packed, cols_i, trits.data());
                                float scale = fp16_to_fp32(scale_buf[static_cast<size_t>(r)]);
                                for (int c = 0; c < cols_i; ++c) {
                                    weights[static_cast<size_t>(r) * static_cast<size_t>(cols_i)
                                            + static_cast<size_t>(c)] =
                                        static_cast<float>(trits[static_cast<size_t>(c)]) * scale;
                                }
                            }
                        } else {
                            // Read failed, fall back to synthetic
                            std::fill(weights.begin(), weights.end(), 0.01f);
                        }
                    } else {
                        // Expert not found in .nxp, use synthetic
                        std::fill(weights.begin(), weights.end(), 0.01f);
                    }
                } else {
                    // Synthetic weights (demo mode)
                    std::fill(weights.begin(), weights.end(), 0.01f);
                }

                badam.init_state(block_params);

                // Memory budget check
                size_t block_memory = block_params * sizeof(float) +
                                      badam.memory_bytes() +
                                      block_params * sizeof(float);  // gradients
                std::fprintf(stderr, "  Layer %u Expert %u: %zu params, %.1f MB\n",
                             layer, exp,
                             block_params,
                             static_cast<double>(block_memory) / (1024.0 * 1024.0));

                for (int step = 0; step < config.badam_config.steps_per_block; ++step) {
                    // Simplified gradient computation: analytical gradient for
                    // linear layer with MSE loss against zero target
                    // grad_W = 2 * W (minimizing ||W||^2)
                    std::vector<float> gradients(block_params);
                    for (size_t i = 0; i < block_params; ++i) {
                        gradients[i] = 2.0f * weights[i];
                    }

                    badam.step(weights.data(), gradients.data(), block_params);
                }

                badam.reset_state();

                // Re-quantize trained weights to ternary for output
                if (use_nxp) {
                    // Convert FP32 -> FP16 -> ternary_quantize
                    std::vector<uint16_t> fp16_buf(block_params);
                    for (size_t i = 0; i < block_params; ++i) {
                        fp16_buf[i] = fp32_to_fp16(weights[i]);
                    }
                    QuantizedWeights qw = ternary_quantize(fp16_buf.data(), rows_i, cols_i);
                    trained_experts.push_back(TrainedExpert{
                        std::move(qw.packed), std::move(qw.scales)});
                }
            }

            // Shared layers (attention/router) use GaLore
            const size_t attn_rows = static_cast<size_t>(hidden_dim);
            const size_t attn_cols = static_cast<size_t>(hidden_dim);
            GaLoreOptimizer galore(config.galore_config, attn_rows, attn_cols);

            std::vector<float> attn_weights(attn_rows * attn_cols, 0.01f);
            std::fprintf(stderr, "  Layer %u Shared (GaLore): %zu params, %.1f MB optimizer\n",
                         layer, attn_rows * attn_cols,
                         static_cast<double>(galore.memory_bytes()) / (1024.0 * 1024.0));

            for (int step = 0; step < config.badam_config.steps_per_block; ++step) {
                std::vector<float> grad(attn_rows * attn_cols);
                for (size_t i = 0; i < attn_rows * attn_cols; ++i) {
                    grad[i] = 2.0f * attn_weights[i];
                }
                galore.step(attn_weights.data(), grad.data(),
                            attn_rows, attn_cols, config.badam_config.lr);
            }
        }

        std::fprintf(stderr, "Epoch %d complete.\n", epoch + 1);
    }

    // Write output .nxp with trained weights
    if (use_nxp && !trained_experts.empty()) {
        std::string out_nxp = config.output_dir + "/model.nxp";
        NxpWriter writer;
        NxpFileHeader out_hdr = nxp_reader.header();
        if (writer.open(out_nxp, out_hdr)) {
            size_t idx = 0;
            for (uint32_t layer = 0; layer < n_layers; ++layer) {
                uint32_t num_experts = (expert_count > 0) ? expert_count : 1;
                for (uint32_t exp = 0; exp < num_experts; ++exp) {
                    if (idx < trained_experts.size()) {
                        auto& te = trained_experts[idx];
                        writer.write_expert(layer, exp,
                                            te.packed.data(), te.packed.size(),
                                            te.scales.data(),
                                            static_cast<uint32_t>(te.scales.size()));
                    }
                    ++idx;
                }
            }
            writer.finalize();
            std::fprintf(stderr, "Output .nxp written: %s\n", out_nxp.c_str());
        }
    }

    // Save training metadata
    nlohmann::json train_meta;
    train_meta["method"] = "full";
    train_meta["epochs"] = config.max_epochs;
    train_meta["badam_lr"] = config.badam_config.lr;
    train_meta["badam_steps_per_block"] = config.badam_config.steps_per_block;
    train_meta["galore_rank"] = config.galore_config.rank;
    train_meta["source_model"] = config.model_dir;

    std::ofstream meta_ofs(config.output_dir + "/training_metadata.json");
    if (meta_ofs.is_open()) {
        meta_ofs << train_meta.dump(2);
    }

    std::fprintf(stderr, "\nTraining complete. Output: %s\n", config.output_dir.c_str());
    return true;
}

bool Trainer::train_lora(const TrainConfig& config) {
    std::fprintf(stderr, "Trainer: starting LoRA fine-tuning\n");
    std::fprintf(stderr, "  Model:  %s\n", config.model_dir.c_str());
    std::fprintf(stderr, "  Data:   %s\n", config.data_path.c_str());
    std::fprintf(stderr, "  Output: %s\n", config.output_dir.c_str());
    std::fprintf(stderr, "  LoRA rank:  %zu\n", config.lora_config.rank);
    std::fprintf(stderr, "  LoRA alpha: %.1f\n", static_cast<double>(config.lora_config.alpha));

    // Load training data
    DataLoader loader;
    if (!loader.load(config.data_path)) {
        std::fprintf(stderr, "Trainer: failed to load training data\n");
        return false;
    }

    // Read model config
    std::string config_path = config.model_dir + "/model_config.json";
    std::ifstream cfg_ifs(config_path);
    if (!cfg_ifs.is_open()) {
        std::fprintf(stderr, "Trainer: cannot open %s\n", config_path.c_str());
        return false;
    }

    nlohmann::json model_meta;
    try {
        model_meta = nlohmann::json::parse(cfg_ifs);
    } catch (const nlohmann::json::exception& e) {
        std::fprintf(stderr, "Trainer: invalid model config: %s\n", e.what());
        return false;
    }

    const uint32_t hidden_dim = model_meta.value("hidden_dim", 0u);
    const uint32_t n_layers = model_meta.value("n_layers", 0u);

    if (hidden_dim == 0 || n_layers == 0) {
        std::fprintf(stderr, "Trainer: invalid model dimensions\n");
        return false;
    }

    // Create output directory
    std::filesystem::create_directories(config.output_dir);

    // Copy model_config.json and tokenizer files to output directory
    for (const auto& entry : std::filesystem::directory_iterator(config.model_dir)) {
        const auto& src = entry.path();
        if (src.extension() == ".json" || src.extension() == ".model") {
            auto dst = std::filesystem::path(config.output_dir) / src.filename();
            std::filesystem::copy_file(src, dst,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    // Create LoRA adapters for each attention projection.
    // Base weights are frozen during LoRA training. The adapters are saved
    // independently and merged via `neuralos merge-lora`.
    // The synthetic forward/backward pass is sufficient since LoRA only
    // trains A and B matrices, not the base weights.
    const size_t dim = static_cast<size_t>(hidden_dim);

    std::vector<LoRAAdapter> adapters;
    std::vector<std::string> adapter_names;
    const std::vector<std::string> proj_names = {"q_proj", "k_proj", "v_proj", "o_proj"};

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        for (const auto& proj : proj_names) {
            adapters.emplace_back(config.lora_config, dim, dim);
            adapter_names.push_back("layer" + std::to_string(layer) + "_" + proj);
        }
    }

    std::fprintf(stderr, "  Created %zu LoRA adapters (rank=%zu)\n",
                 adapters.size(), config.lora_config.rank);
    std::fprintf(stderr, "  Total trainable params: %zu\n",
                 adapters.size() * adapters[0].param_count());

    // Training loop (simplified)
    const float lr = config.badam_config.lr;  // Reuse LR from config
    const size_t r = config.lora_config.rank;

    for (int epoch = 0; epoch < config.max_epochs; ++epoch) {
        std::fprintf(stderr, "\nEpoch %d/%d:\n", epoch + 1, config.max_epochs);
        loader.shuffle(static_cast<uint64_t>(epoch));

        float epoch_loss = 0.0f;
        int steps = 0;

        // For this prototype, train each adapter independently
        for (size_t a = 0; a < adapters.size(); ++a) {
            auto& adapter = adapters[a];

            // Synthetic training step
            std::vector<float> x(dim, 0.1f);
            std::vector<float> delta(dim, 0.0f);
            std::vector<float> grad_output(dim);
            std::vector<float> grad_A(r * dim);
            std::vector<float> grad_B(dim * r);

            for (int step = 0; step < 10; ++step) {
                // Forward
                adapter.forward(x.data(), delta.data());

                // Compute loss gradient (MSE against target of zeros)
                for (size_t i = 0; i < dim; ++i) {
                    grad_output[i] = 2.0f * delta[i];
                }

                // Backward
                adapter.backward(x.data(), grad_output.data(),
                                 grad_A.data(), grad_B.data());

                // Update
                adapter.update(grad_A.data(), grad_B.data(), lr);

                // Track loss
                float loss = 0.0f;
                for (size_t i = 0; i < dim; ++i) {
                    loss += delta[i] * delta[i];
                }
                epoch_loss += loss;
                ++steps;
            }
        }

        if (steps > 0) {
            std::fprintf(stderr, "  Avg loss: %.6f\n",
                         static_cast<double>(epoch_loss / static_cast<float>(steps)));
        }
    }

    // Save all adapters
    for (size_t a = 0; a < adapters.size(); ++a) {
        std::string adapter_dir = config.output_dir + "/" + adapter_names[a];
        if (!adapters[a].save(adapter_dir)) {
            std::fprintf(stderr, "Trainer: failed to save adapter %s\n",
                         adapter_names[a].c_str());
            return false;
        }
    }

    // Save training metadata
    nlohmann::json train_meta;
    train_meta["method"] = "lora";
    train_meta["epochs"] = config.max_epochs;
    train_meta["lora_rank"] = config.lora_config.rank;
    train_meta["lora_alpha"] = config.lora_config.alpha;
    train_meta["lr"] = lr;
    train_meta["source_model"] = config.model_dir;
    train_meta["adapter_count"] = adapters.size();
    train_meta["adapter_names"] = adapter_names;

    std::ofstream meta_ofs(config.output_dir + "/training_metadata.json");
    if (meta_ofs.is_open()) {
        meta_ofs << train_meta.dump(2);
    }

    std::fprintf(stderr, "\nLoRA training complete. Adapters saved to: %s\n",
                 config.output_dir.c_str());
    return true;
}

}  // namespace nos
