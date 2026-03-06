#include "engine/router.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace nos {

void Router::load(const float* weights, int num_experts, int hidden_dim) {
    weights_ = weights;
    num_experts_ = num_experts;
    hidden_dim_ = hidden_dim;
}

RouterResult Router::route(const float* hidden_state, int k) const {
    auto sz = [](int x) -> size_t { return static_cast<size_t>(x); };

    std::vector<float> scores(sz(num_experts_));
    for (size_t e = 0; e < sz(num_experts_); e++) {
        float dot = 0.0f;
        const float* row = weights_ + e * sz(hidden_dim_);
        for (int d = 0; d < hidden_dim_; d++) {
            dot += row[d] * hidden_state[d];
        }
        scores[e] = dot;
    }

    std::vector<size_t> indices(sz(num_experts_));
    std::iota(indices.begin(), indices.end(), size_t{0});
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

    RouterResult result;
    result.expert_ids.resize(sz(k));
    result.gates.resize(sz(k));
    result.raw_scores = scores;

    float max_score = scores[indices[0]];
    float sum_exp = 0.0f;
    for (size_t i = 0; i < sz(k); i++) {
        result.gates[i] = std::exp(scores[indices[i]] - max_score);
        sum_exp += result.gates[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < sz(k); i++) {
        result.expert_ids[i] = static_cast<uint32_t>(indices[i]);
        result.gates[i] *= inv_sum;
    }

    return result;
}

}  // namespace nos
