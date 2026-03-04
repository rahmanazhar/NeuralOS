#pragma once

/// @file inference_engine.h
/// @brief Full transformer forward pass with VMM-integrated expert loading.
///
/// Loads a converted .nxp model via shared ModelConfig, performs the complete
/// Llama-family transformer forward pass (embedding, RMSNorm, RoPE, attention,
/// MoE routing, ternary expert compute, output projection).

#include "converter/model_config.h"
#include "engine/attention.h"
#include "engine/rope.h"
#include "engine/router.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nos {

class Vmm;

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /// Load model from converted .nxp directory.
    /// Reads model_config.json via nos::from_json() into nos::ModelConfig.
    /// @param model_dir    Directory containing model.nxp and model_config.json
    /// @param vmm          VMM instance for expert paging (non-owning)
    /// @param num_threads  Thread count for expert-parallel dispatch (0 = auto)
    bool load(const std::string& model_dir, Vmm* vmm, int num_threads = 0);

    /// Forward pass for a single token at position pos.
    /// Returns pointer to logits buffer (vocab_size floats).
    const float* forward_step(int token_id, int pos);

    /// Reset KV cache (for new sequence).
    void reset_kv_cache();

    int vocab_size() const;
    const ModelConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nos
