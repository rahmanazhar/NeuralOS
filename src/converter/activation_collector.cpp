/// @file activation_collector.cpp
/// @brief Activation magnitude collection implementation.

#include "converter/activation_collector.h"
#include "converter/model_reader.h"
#include "kernel/packing.h"  // fp16_to_fp32, fp32_to_fp16

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

namespace nos {

void dense_matvec_fp16(const uint16_t* W, const float* x, float* out,
                        int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint16_t* row = W + r * cols;
        for (int c = 0; c < cols; c++) {
            acc += fp16_to_fp32(row[c]) * x[c];
        }
        out[r] = acc;
    }
}

namespace {

/// SiLU activation function: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

/// Simple byte-level tokenizer: split text into byte tokens (0-255).
/// Sufficient for activation clustering -- we only need approximate embeddings.
std::vector<uint32_t> byte_tokenize(const std::string& text, int max_tokens) {
    std::vector<uint32_t> tokens;
    tokens.reserve(static_cast<size_t>(
        std::min(static_cast<int>(text.size()), max_tokens)));
    for (size_t i = 0; i < text.size() && static_cast<int>(tokens.size()) < max_tokens; i++) {
        tokens.push_back(static_cast<uint32_t>(static_cast<uint8_t>(text[i])));
    }
    return tokens;
}

/// Load calibration text from file.
std::string load_calibration_text(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    return std::string(std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>());
}

/// Look up embedding vectors for token IDs.
/// Returns [n_tokens x hidden_dim] float matrix.
std::vector<float> lookup_embeddings(ModelReader& reader,
                                      const std::vector<uint32_t>& token_ids,
                                      int hidden_dim) {
    // Find the embedding tensor
    const TensorInfo* emb_info = nullptr;
    std::vector<std::string> names = {"model.embed_tokens.weight",
                                       "token_embd.weight",
                                       "tok_embeddings.weight"};
    for (const auto& name : names) {
        emb_info = reader.find_tensor(name);
        if (emb_info) break;
    }

    std::vector<float> embeddings(token_ids.size() * static_cast<size_t>(hidden_dim), 0.0f);

    if (!emb_info) {
        // No embedding tensor found -- return zeros (test fallback)
        return embeddings;
    }

    // Read embeddings row by row
    size_t elem_size = emb_info->element_size();
    std::vector<uint8_t> row_buf(static_cast<size_t>(hidden_dim) * elem_size);

    for (size_t i = 0; i < token_ids.size(); i++) {
        uint32_t tid = token_ids[i];
        if (static_cast<int64_t>(tid) >= emb_info->shape[0]) {
            // Token ID out of range -- use zeros
            continue;
        }

        if (reader.read_tensor_rows(*emb_info, static_cast<int64_t>(tid), 1,
                                     row_buf.data(), row_buf.size())) {
            float* out = &embeddings[i * static_cast<size_t>(hidden_dim)];
            if (emb_info->dtype == "F16") {
                const auto* fp16 = reinterpret_cast<const uint16_t*>(row_buf.data());
                for (int d = 0; d < hidden_dim; d++) {
                    out[d] = fp16_to_fp32(fp16[d]);
                }
            } else if (emb_info->dtype == "F32") {
                std::memcpy(out, row_buf.data(),
                            static_cast<size_t>(hidden_dim) * sizeof(float));
            }
        }
    }

    return embeddings;
}

}  // namespace

ActivationData collect_activations(ModelReader& reader,
                                    const ActivationCollectorConfig& config) {
    ActivationData result;

    auto model_cfg = reader.config();
    int n_layers = static_cast<int>(model_cfg.n_layers);
    int hidden_dim = static_cast<int>(model_cfg.hidden_dim);
    int intermediate_dim = static_cast<int>(model_cfg.intermediate_dim);

    if (n_layers == 0 || hidden_dim == 0 || intermediate_dim == 0) {
        return result;
    }

    // --- Prepare calibration inputs ---
    std::vector<uint32_t> token_ids;

    if (!config.calibration_data_path.empty()) {
        // Load real calibration text
        std::string text = load_calibration_text(config.calibration_data_path);
        if (!text.empty()) {
            token_ids = byte_tokenize(text, config.calibration_samples);
        }
    }

    if (token_ids.empty()) {
        // Synthetic mode: generate sequential token IDs
        if (!config.calibration_data_path.empty()) {
            std::fprintf(stderr, "WARNING: Could not load calibration data from '%s'\n",
                        config.calibration_data_path.c_str());
        }
        std::fprintf(stderr,
            "WARNING: No calibration data path provided, using synthetic embeddings "
            "(test mode only)\n");

        int n_samples = std::min(config.calibration_samples,
                                  static_cast<int>(model_cfg.vocab_size));
        if (n_samples <= 0) n_samples = 64;
        token_ids.resize(static_cast<size_t>(n_samples));
        std::iota(token_ids.begin(), token_ids.end(), 0u);
    }

    // Look up embeddings from model
    auto embeddings = lookup_embeddings(reader, token_ids, hidden_dim);
    int n_samples = static_cast<int>(token_ids.size());

    result.per_layer_magnitudes.resize(static_cast<size_t>(n_layers));

    // --- Process each layer ---
    for (int layer = 0; layer < n_layers; layer++) {
        // Initialize per-neuron magnitude accumulator
        std::vector<float> neuron_magnitudes(static_cast<size_t>(intermediate_dim), 0.0f);

        // Try to load gate_proj weights for this layer
        std::vector<std::string> gate_names = {
            "model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight",
            "blk." + std::to_string(layer) + ".ffn_gate.weight"
        };

        const TensorInfo* gate_info = nullptr;
        for (const auto& name : gate_names) {
            gate_info = reader.find_tensor(name);
            if (gate_info) break;
        }

        if (!gate_info) {
            // No gate tensor found -- use uniform magnitudes (fallback for test)
            std::fill(neuron_magnitudes.begin(), neuron_magnitudes.end(), 1.0f);
            result.per_layer_magnitudes[static_cast<size_t>(layer)] = std::move(neuron_magnitudes);
            continue;
        }

        // Load gate weights [intermediate_dim x hidden_dim] as FP16
        size_t gate_elems = static_cast<size_t>(intermediate_dim) * static_cast<size_t>(hidden_dim);
        std::vector<uint16_t> gate_weights(gate_elems);

        if (gate_info->dtype == "F16") {
            reader.read_tensor(*gate_info, gate_weights.data(),
                              gate_elems * sizeof(uint16_t));
        } else if (gate_info->dtype == "F32") {
            // Convert F32 to F16
            std::vector<float> f32_buf(gate_elems);
            reader.read_tensor(*gate_info, f32_buf.data(), gate_elems * sizeof(float));
            for (size_t i = 0; i < gate_elems; i++) {
                gate_weights[i] = fp32_to_fp16(f32_buf[i]);
            }
        }

        // Run calibration samples through gate_proj and accumulate neuron magnitudes
        std::vector<float> gate_output(static_cast<size_t>(intermediate_dim));

        for (int s = 0; s < n_samples; s++) {
            const float* input = &embeddings[static_cast<size_t>(s) * static_cast<size_t>(hidden_dim)];

            // gate_proj: [intermediate_dim x hidden_dim] * [hidden_dim] -> [intermediate_dim]
            dense_matvec_fp16(gate_weights.data(), input, gate_output.data(),
                              intermediate_dim, hidden_dim);

            // Apply SiLU and accumulate squared magnitudes
            for (int n = 0; n < intermediate_dim; n++) {
                float activated = silu(gate_output[static_cast<size_t>(n)]);
                neuron_magnitudes[static_cast<size_t>(n)] += activated * activated;
            }
        }

        // Convert to L2 norms
        for (int n = 0; n < intermediate_dim; n++) {
            neuron_magnitudes[static_cast<size_t>(n)] =
                std::sqrt(neuron_magnitudes[static_cast<size_t>(n)] / static_cast<float>(n_samples));
        }

        result.per_layer_magnitudes[static_cast<size_t>(layer)] = std::move(neuron_magnitudes);
    }

    return result;
}

}  // namespace nos
