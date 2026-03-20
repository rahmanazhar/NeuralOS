/// @file test_badam.cpp
/// @brief Catch2 tests for BAdam block-wise Adam optimizer.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "training/badam.h"

#include <cmath>
#include <vector>

TEST_CASE("AdamW converges on x^2", "[badam]") {
    // Minimize f(x) = x^2. Gradient = 2x.
    // After enough steps, x should be near 0.
    nos::BAdamConfig cfg;
    cfg.lr = 0.05f;
    cfg.weight_decay = 0.0f;  // Pure Adam, no weight decay for this test

    nos::BAdamOptimizer opt(cfg);

    float x = 5.0f;
    opt.init_state(1);

    for (int i = 0; i < 500; ++i) {
        float grad = 2.0f * x;
        opt.step(&x, &grad, 1);
    }

    REQUIRE(std::fabs(x) < 0.5f);
}

TEST_CASE("AdamW weight decay shrinks weights", "[badam]") {
    nos::BAdamConfig cfg;
    cfg.lr = 0.01f;
    cfg.weight_decay = 0.1f;

    nos::BAdamOptimizer opt(cfg);

    // Start with large weight, zero gradients
    float w = 10.0f;
    opt.init_state(1);

    float initial = w;
    float grad = 0.0f;
    for (int i = 0; i < 50; ++i) {
        opt.step(&w, &grad, 1);
    }

    // Weight should shrink due to decay
    REQUIRE(std::fabs(w) < std::fabs(initial));
}

TEST_CASE("BAdam reset_state zeroes m and v", "[badam]") {
    nos::BAdamConfig cfg;
    nos::BAdamOptimizer opt(cfg);

    opt.init_state(10);

    // Do a few steps to populate m and v
    std::vector<float> w(10, 1.0f);
    std::vector<float> g(10, 0.5f);
    for (int i = 0; i < 5; ++i) {
        opt.step(w.data(), g.data(), 10);
    }

    REQUIRE(opt.step_count() == 5);

    opt.reset_state();
    REQUIRE(opt.step_count() == 0);
    REQUIRE(opt.memory_bytes() == 2 * 10 * sizeof(float));
}

TEST_CASE("BAdam memory_bytes is 2*N*4", "[badam]") {
    nos::BAdamOptimizer opt(nos::BAdamConfig{});

    opt.init_state(1024);
    REQUIRE(opt.memory_bytes() == 2 * 1024 * sizeof(float));

    opt.init_state(256);
    REQUIRE(opt.memory_bytes() == 2 * 256 * sizeof(float));
}

TEST_CASE("BAdam step counter increments correctly", "[badam]") {
    nos::BAdamOptimizer opt(nos::BAdamConfig{});
    opt.init_state(4);

    std::vector<float> w(4, 1.0f);
    std::vector<float> g(4, 0.1f);

    REQUIRE(opt.step_count() == 0);
    opt.step(w.data(), g.data(), 4);
    REQUIRE(opt.step_count() == 1);
    opt.step(w.data(), g.data(), 4);
    REQUIRE(opt.step_count() == 2);
    opt.step(w.data(), g.data(), 4);
    REQUIRE(opt.step_count() == 3);
}

TEST_CASE("BAdam multi-parameter convergence", "[badam]") {
    // Minimize f(x,y) = x^2 + y^2
    nos::BAdamConfig cfg;
    cfg.lr = 0.05f;
    cfg.weight_decay = 0.0f;

    nos::BAdamOptimizer opt(cfg);

    std::vector<float> w = {3.0f, -4.0f};
    opt.init_state(2);

    for (int i = 0; i < 500; ++i) {
        std::vector<float> g = {2.0f * w[0], 2.0f * w[1]};
        opt.step(w.data(), g.data(), 2);
    }

    REQUIRE(std::fabs(w[0]) < 0.5f);
    REQUIRE(std::fabs(w[1]) < 0.5f);
}

TEST_CASE("BAdam config access", "[badam]") {
    nos::BAdamConfig cfg;
    cfg.lr = 0.001f;
    cfg.beta1 = 0.85f;
    cfg.steps_per_block = 50;

    nos::BAdamOptimizer opt(cfg);
    REQUIRE(opt.config().lr == 0.001f);
    REQUIRE(opt.config().beta1 == 0.85f);
    REQUIRE(opt.config().steps_per_block == 50);
}
