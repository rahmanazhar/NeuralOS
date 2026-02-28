#pragma once

/// @file model_config.h
/// @brief Shared model configuration struct with JSON serialization.
///
/// This is the SINGLE SOURCE OF TRUTH for the model_config.json schema.
/// Both the conversion pipeline (writer) and inference engine (reader)
/// use this struct to ensure schema agreement.

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace nos {

struct ModelConfig {
    // Architecture identification
    std::string architecture = "llama";  // "llama", "mistral", etc.

    // Core dimensions
    uint32_t vocab_size = 0;
    uint32_t hidden_dim = 0;
    uint32_t intermediate_dim = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;       // == n_heads for MHA, < n_heads for GQA, == 1 for MQA
    uint32_t head_dim = 0;         // typically hidden_dim / n_heads
    uint32_t max_seq_len = 2048;   // context length

    // Normalization / position encoding
    float rope_theta = 10000.0f;
    float norm_eps = 1e-5f;        // rms_norm_eps

    // MoE configuration (set by conversion pipeline)
    uint32_t expert_count = 0;     // experts per layer (0 = dense model, not yet converted)
    uint32_t top_k = 2;            // experts selected per token
    uint32_t max_expert_size = 0;  // bytes, largest expert slab

    // Attention type (derived from n_heads vs n_kv_heads)
    std::string attention_type = "mha";  // "mha", "gqa", "mqa"
};

// JSON serialization -- defines the model_config.json schema
inline void to_json(nlohmann::json& j, const ModelConfig& c) {
    j = nlohmann::json{
        {"architecture", c.architecture},
        {"vocab_size", c.vocab_size},
        {"hidden_dim", c.hidden_dim},
        {"intermediate_dim", c.intermediate_dim},
        {"n_layers", c.n_layers},
        {"n_heads", c.n_heads},
        {"n_kv_heads", c.n_kv_heads},
        {"head_dim", c.head_dim},
        {"max_seq_len", c.max_seq_len},
        {"rope_theta", c.rope_theta},
        {"norm_eps", c.norm_eps},
        {"expert_count", c.expert_count},
        {"top_k", c.top_k},
        {"max_expert_size", c.max_expert_size},
        {"attention_type", c.attention_type}
    };
}

inline void from_json(const nlohmann::json& j, ModelConfig& c) {
    j.at("architecture").get_to(c.architecture);
    j.at("vocab_size").get_to(c.vocab_size);
    j.at("hidden_dim").get_to(c.hidden_dim);
    j.at("intermediate_dim").get_to(c.intermediate_dim);
    j.at("n_layers").get_to(c.n_layers);
    j.at("n_heads").get_to(c.n_heads);
    j.at("n_kv_heads").get_to(c.n_kv_heads);
    j.at("head_dim").get_to(c.head_dim);
    j.at("max_seq_len").get_to(c.max_seq_len);
    j.at("rope_theta").get_to(c.rope_theta);
    j.at("norm_eps").get_to(c.norm_eps);
    j.at("expert_count").get_to(c.expert_count);
    j.at("top_k").get_to(c.top_k);
    j.at("max_expert_size").get_to(c.max_expert_size);
    j.at("attention_type").get_to(c.attention_type);
}

// Derive attention_type from head counts
inline std::string derive_attention_type(uint32_t n_heads, uint32_t n_kv_heads) {
    if (n_kv_heads == n_heads) return "mha";
    if (n_kv_heads == 1) return "mqa";
    return "gqa";
}

}  // namespace nos
