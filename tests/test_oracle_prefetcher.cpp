/// @file test_oracle_prefetcher.cpp
/// @brief Catch2 tests for OraclePrefetcher integration logic.
///
/// Uses null VMM to avoid real VMM dependency; OraclePrefetcher must handle
/// vmm_=nullptr gracefully by skipping dispatch calls.

#include "engine/benchmark.h"
#include "engine/metrics.h"
#include "engine/oracle_prefetcher.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

// ── Helpers ──────────────────────────────────────────────────────────────────

static nos::OraclePrefetcher::Config make_config(int n_layers = 4, int num_experts = 8,
                                                  int max_k = 3) {
    nos::OraclePrefetcher::Config cfg;
    cfg.n_layers    = n_layers;
    cfg.num_experts = num_experts;
    cfg.max_k       = max_k;
    cfg.lstm_hidden_dim        = 16;
    cfg.lstm_proj_dim          = 8;
    cfg.online_update_interval = 8;
    cfg.cooldown_tokens        = 5;
    return cfg;
}

static std::string slurp(const std::string& path) {
    std::ifstream ifs(path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static nos::VmmStats make_vmm_stats() {
    nos::VmmStats s{};
    s.total_pins   = 200;
    s.cache_hits   = 180;
    s.cache_misses = 20;
    s.evictions    = 5;
    return s;
}

static nos::StickyRouter::AggregateMetrics make_routing() {
    nos::StickyRouter::AggregateMetrics m;
    m.total_routing_decisions = 500;
    m.total_switches          = 25;
    m.switch_rate             = 0.05f;
    m.avg_window_length       = 20.0f;
    return m;
}

// ── Test 1: Construction with null VMM ──────────────────────────────────────

TEST_CASE("OraclePrefetcher construction with null VMM", "[oracle_prefetcher]") {
    nos::OraclePrefetcher::Config cfg = make_config();
    // Must not crash with null vmm and null metrics
    REQUIRE_NOTHROW(nos::OraclePrefetcher(cfg, nullptr, nullptr));
}

// ── Test 2: predict_and_dispatch no crash with null VMM ─────────────────────

TEST_CASE("OraclePrefetcher predict_and_dispatch no crash", "[oracle_prefetcher]") {
    auto cfg = make_config();
    nos::OraclePrefetcher op(cfg, nullptr, nullptr);

    std::vector<float> logits(static_cast<size_t>(cfg.num_experts), 0.1f);
    std::vector<float> hidden(32, 0.5f);
    std::vector<uint32_t> chosen = {0u, 1u};

    // Should not crash even with null vmm (dispatch is skipped)
    REQUIRE_NOTHROW(op.predict_and_dispatch(0, chosen, logits.data(),
                                             hidden.data(),
                                             static_cast<int>(hidden.size()), 2));
}

// ── Test 3: Fallback on K reduction ─────────────────────────────────────────

TEST_CASE("OraclePrefetcher fallback on K reduction", "[oracle_prefetcher]") {
    auto cfg = make_config();
    nos::MetricsCollector metrics;
    metrics.register_defaults();
    nos::OraclePrefetcher op(cfg, nullptr, &metrics);

    std::vector<float> logits(static_cast<size_t>(cfg.num_experts), 0.1f);
    std::vector<float> hidden(16, 0.5f);
    std::vector<uint32_t> chosen = {0u, 1u};

    // Run enough tokens to advance past warmup
    for (int t = 0; t < 70; t++) {
        for (int l = 0; l < cfg.n_layers; l++) {
            op.predict_and_dispatch(l, chosen, logits.data(),
                                    hidden.data(),
                                    static_cast<int>(hidden.size()), 2);
        }
        op.tick(50.0);
    }

    // stats().mode must be a non-empty string
    auto s = op.stats();
    REQUIRE_FALSE(s.mode.empty());
}

// ── Test 4: reset clears state ───────────────────────────────────────────────

TEST_CASE("OraclePrefetcher reset clears state", "[oracle_prefetcher]") {
    auto cfg = make_config();
    nos::OraclePrefetcher op(cfg, nullptr, nullptr);

    std::vector<float> logits(static_cast<size_t>(cfg.num_experts), 0.1f);
    std::vector<float> hidden(16, 0.5f);
    std::vector<uint32_t> chosen = {0u};

    // Run 10 tokens
    for (int t = 0; t < 10; t++) {
        for (int l = 0; l < cfg.n_layers; l++) {
            op.predict_and_dispatch(l, chosen, logits.data(),
                                    hidden.data(),
                                    static_cast<int>(hidden.size()), 1);
        }
        op.tick(50.0);
    }

    // Reset and verify counters are cleared
    op.reset();
    auto s = op.stats();
    REQUIRE(s.speculative_hits   == 0u);
    REQUIRE(s.speculative_misses == 0u);
    // effective_k should be back to max_k after reset
    REQUIRE(s.effective_k == cfg.max_k);
}

// ── Test 5: set_enabled false skips dispatch ─────────────────────────────────

TEST_CASE("OraclePrefetcher set_enabled false skips dispatch", "[oracle_prefetcher]") {
    auto cfg = make_config();
    nos::OraclePrefetcher op(cfg, nullptr, nullptr);

    // Disable immediately
    op.set_enabled(false);
    REQUIRE_FALSE(op.enabled());

    std::vector<float> logits(static_cast<size_t>(cfg.num_experts), 1.0f);
    std::vector<float> hidden(16, 1.0f);
    std::vector<uint32_t> chosen = {0u, 1u};

    for (int t = 0; t < 5; t++) {
        for (int l = 0; l < cfg.n_layers; l++) {
            op.predict_and_dispatch(l, chosen, logits.data(),
                                    hidden.data(),
                                    static_cast<int>(hidden.size()), 2);
        }
        op.tick(50.0);
    }

    auto s = op.stats();
    // When disabled, no dispatch happens, so hits+misses stays 0
    REQUIRE(s.speculative_hits   == 0u);
    REQUIRE(s.speculative_misses == 0u);
    REQUIRE(s.mode == "none");
}

// ── Test 6: PrefetchStats default mode is "none" ────────────────────────────

TEST_CASE("PrefetchStats default mode is none", "[oracle_prefetcher]") {
    nos::PrefetchStats s;
    REQUIRE(s.mode == "none");
    REQUIRE(s.effective_k == 0);
    REQUIRE(s.speculative_hits   == 0u);
    REQUIRE(s.speculative_misses == 0u);
    REQUIRE(s.rwp_oracle       == 0.0);
    REQUIRE(s.rwp_best_baseline == 0.0);
}

// ── Test 7: BenchmarkReporter write_csv includes prefetch_mode ───────────────

TEST_CASE("BenchmarkReporter write_csv includes prefetch_mode", "[oracle_prefetcher]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_prefetch_csv";
    std::filesystem::create_directories(tmp);
    std::string csv_path = (tmp / "test.csv").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    metrics.register_defaults();
    metrics.inc_counter("tokens_generated", 100);
    metrics.inc_counter("expert_reuses", 80);
    metrics.inc_counter("expert_loads", 20);
    for (int i = 0; i < 100; i++) {
        metrics.observe_histogram("token_latency_ms", 10.0 + static_cast<double>(i % 20));
    }

    nos::PrefetchStats ps;
    ps.mode        = "oracle";
    ps.effective_k = 5;

    reporter.write_csv(csv_path, metrics, make_vmm_stats(), make_routing(), ps);

    auto content = slurp(csv_path);
    REQUIRE_THAT(content, ContainsSubstring("prefetch_mode"));
    REQUIRE_THAT(content, ContainsSubstring("oracle"));
    REQUIRE_THAT(content, ContainsSubstring("5"));

    std::filesystem::remove_all(tmp);
}

// ── Test 8: BenchmarkReporter write_json includes "prefetch" object ──────────

TEST_CASE("BenchmarkReporter write_json includes prefetch object", "[oracle_prefetcher]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_prefetch_json";
    std::filesystem::create_directories(tmp);
    std::string json_path = (tmp / "test.json").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    metrics.register_defaults();
    metrics.inc_counter("tokens_generated", 100);
    metrics.inc_counter("expert_reuses", 80);
    metrics.inc_counter("expert_loads", 20);
    for (int i = 0; i < 100; i++) {
        metrics.observe_histogram("token_latency_ms", 10.0 + static_cast<double>(i % 20));
    }

    nos::PrefetchStats ps;
    ps.mode        = "oracle";
    ps.effective_k = 5;

    reporter.write_json(json_path, metrics, make_vmm_stats(), make_routing(), ps);

    auto content = slurp(json_path);
    auto j = nlohmann::json::parse(content);

    REQUIRE(j.contains("prefetch"));
    REQUIRE(j["prefetch"]["mode"] == "oracle");
    REQUIRE(j["prefetch"]["effective_k"] == 5);

    std::filesystem::remove_all(tmp);
}
