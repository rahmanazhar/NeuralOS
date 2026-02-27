#pragma once

/// @file memory_budget.h
/// @brief Memory budget partitioning for the Virtual Memory Manager.
///
/// The user specifies --memory budget (e.g., 8G). compute_budget() deterministically
/// partitions the budget into expert cache, KV cache, working buffers, and OS overhead.
/// The system refuses to start if the budget is insufficient.

#include <cstddef>
#include <cstdint>
#include <string>

namespace nos {

/// Model-specific parameters needed for budget computation.
struct ModelParams {
    uint32_t n_layers;            ///< Total model layers
    uint32_t n_kv_heads;          ///< KV attention heads per layer
    uint32_t head_dim;            ///< Dimension per attention head
    uint32_t hidden_dim;          ///< Model hidden dimension
    uint32_t max_expert_size;     ///< Largest expert weight+scale bytes
    uint32_t top_k;               ///< Number of experts activated per token
    uint32_t experts_per_layer;   ///< Experts per MoE layer
};

/// Result of budget computation.
struct BudgetPartition {
    size_t   total;              ///< User --memory budget in bytes
    size_t   expert_cache;       ///< Bytes for expert slab pool
    size_t   kv_cache;           ///< Bytes reserved for KV cache
    size_t   working_buffers;    ///< Bytes for activations and scratch
    size_t   os_overhead;        ///< Reserved for runtime, stacks, index
    uint32_t max_context;        ///< Maximum context tokens supported
    uint32_t expert_slots;       ///< Number of slab slots available
    bool     sufficient;         ///< true if budget meets minimum requirements
    size_t   minimum_required;   ///< If !sufficient, the minimum bytes needed
};

/// Compute a deterministic budget partition from the user's memory budget.
///
/// @param user_budget_bytes  Total memory budget in bytes (e.g., 8GB)
/// @param params             Model parameters (layers, heads, expert sizes)
/// @param desired_context_length  Desired KV context length in tokens
/// @return BudgetPartition with all fields computed
BudgetPartition compute_budget(size_t user_budget_bytes,
                               const ModelParams& params,
                               uint32_t desired_context_length);

/// Format a human-readable budget report for stderr.
std::string format_budget_report(const BudgetPartition& bp,
                                 const ModelParams& params);

/// Format bytes as "X.XX GB" or "X.XX MB".
std::string format_bytes(size_t bytes);

/// Parse memory string like "8G", "8GB", "512M", "512MB" into bytes.
/// Case-insensitive. Returns 0 on parse failure.
size_t parse_memory_string(const std::string& str);

}  // namespace nos
