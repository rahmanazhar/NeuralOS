#pragma once

/// @file router.h
/// @brief Top-k expert routing with softmax-renormalized gating.

#include <cstdint>
#include <vector>

namespace nos {

/// Result of top-k expert routing.
struct RouterResult {
    std::vector<uint32_t> expert_ids;  ///< k selected expert indices
    std::vector<float> gates;          ///< k gating weights (sum to 1.0)
};

/// Top-k expert router for MoE layers.
class Router {
public:
    /// Load router weights for one layer.
    ///
    /// @param weights     Router weight matrix [num_experts x hidden_dim], non-owning
    /// @param num_experts Number of experts in this layer
    /// @param hidden_dim  Hidden state dimension
    void load(const float* weights, int num_experts, int hidden_dim);

    /// Select top-k experts for a hidden state.
    ///
    /// Computes dot products, selects top-k, applies softmax renormalization.
    ///
    /// @param hidden_state  Input hidden state [hidden_dim]
    /// @param k             Number of experts to select
    /// @return RouterResult with selected expert IDs and normalized gating weights
    RouterResult route(const float* hidden_state, int k) const;

    int num_experts() const { return num_experts_; }
    int hidden_dim() const { return hidden_dim_; }

private:
    const float* weights_ = nullptr;  // Non-owning
    int num_experts_ = 0;
    int hidden_dim_ = 0;
};

}  // namespace nos
