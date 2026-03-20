/// @file badam.cpp
/// @brief BAdam block-wise Adam optimizer implementation.

#include "training/badam.h"

#include <cmath>
#include <cstring>

namespace nos {

BAdamOptimizer::BAdamOptimizer(BAdamConfig config)
    : config_(config) {}

void BAdamOptimizer::init_state(size_t num_params) {
    state_.m.assign(num_params, 0.0f);
    state_.v.assign(num_params, 0.0f);
    state_.step = 0;
}

void BAdamOptimizer::step(float* weights, const float* gradients,
                          size_t num_params) {
    ++state_.step;

    const float beta1 = config_.beta1;
    const float beta2 = config_.beta2;
    const float lr = config_.lr;
    const float eps = config_.eps;
    const float wd = config_.weight_decay;

    // Bias correction denominators
    const float t = static_cast<float>(state_.step);
    const float bc1 = 1.0f - std::pow(beta1, t);
    const float bc2 = 1.0f - std::pow(beta2, t);

    for (size_t i = 0; i < num_params; ++i) {
        const float g = gradients[i];

        // Update biased first moment estimate
        state_.m[i] = beta1 * state_.m[i] + (1.0f - beta1) * g;

        // Update biased second raw moment estimate
        state_.v[i] = beta2 * state_.v[i] + (1.0f - beta2) * g * g;

        // Bias-corrected estimates
        const float m_hat = state_.m[i] / bc1;
        const float v_hat = state_.v[i] / bc2;

        // AdamW: weight decay applied directly to weights (decoupled)
        weights[i] = weights[i] * (1.0f - lr * wd)
                   - lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void BAdamOptimizer::reset_state() {
    std::fill(state_.m.begin(), state_.m.end(), 0.0f);
    std::fill(state_.v.begin(), state_.v.end(), 0.0f);
    state_.step = 0;
}

size_t BAdamOptimizer::memory_bytes() const {
    return 2 * state_.m.size() * sizeof(float);
}

size_t BAdamOptimizer::step_count() const {
    return state_.step;
}

const BAdamConfig& BAdamOptimizer::config() const {
    return config_;
}

}  // namespace nos
