/// @file test_benchmark_output.cpp
/// @brief Tests for BenchmarkReporter (CSV, JSON, LaTeX output).

#include "engine/benchmark.h"
#include "engine/metrics.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Catch::Matchers::ContainsSubstring;

// Helper to read a file into a string.
static std::string slurp(const std::string& path) {
    std::ifstream ifs(path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Helper to populate a MetricsCollector for tests (non-copyable due to mutex).
static void populate_test_metrics(nos::MetricsCollector& mc) {
    mc.register_defaults();
    mc.inc_counter("tokens_generated", 100);
    mc.inc_counter("expert_reuses", 80);
    mc.inc_counter("expert_loads", 20);
    mc.inc_counter("shift_detections", 3);
    for (int i = 0; i < 100; i++) {
        mc.observe_histogram("token_latency_ms", 10.0 + static_cast<double>(i % 20));
    }
}

static nos::VmmStats make_test_vmm_stats() {
    nos::VmmStats s{};
    s.total_pins = 200;
    s.cache_hits = 180;
    s.cache_misses = 20;
    s.evictions = 5;
    s.crc_failures = 0;
    return s;
}

static nos::StickyRouter::AggregateMetrics make_test_routing() {
    nos::StickyRouter::AggregateMetrics m;
    m.total_routing_decisions = 500;
    m.total_switches = 25;
    m.switch_rate = 0.05f;
    m.avg_window_length = 20.0f;
    return m;
}

TEST_CASE("CSV output has correct headers and data row", "[benchmark]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_test_csv";
    std::filesystem::create_directories(tmp);
    std::string csv_path = (tmp / "test.csv").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    populate_test_metrics(metrics);
    reporter.write_csv(csv_path, metrics, make_test_vmm_stats(), make_test_routing());

    auto content = slurp(csv_path);
    REQUIRE_THAT(content, ContainsSubstring("model,tokens,total_time_ms"));
    REQUIRE_THAT(content, ContainsSubstring("test_model"));
    REQUIRE_THAT(content, ContainsSubstring("100,5000.0"));

    std::filesystem::remove_all(tmp);
}

TEST_CASE("JSON output parses as valid JSON with expected fields", "[benchmark]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_test_json";
    std::filesystem::create_directories(tmp);
    std::string json_path = (tmp / "test.json").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    populate_test_metrics(metrics);
    reporter.write_json(json_path, metrics, make_test_vmm_stats(), make_test_routing());

    auto content = slurp(json_path);
    auto j = nlohmann::json::parse(content);

    REQUIRE(j.contains("run_info"));
    REQUIRE(j["run_info"]["model"] == "test_model");
    REQUIRE(j["run_info"]["tokens_generated"] == 100);
    REQUIRE(j.contains("vmm_stats"));
    REQUIRE(j["vmm_stats"]["cache_hits"] == 180);
    REQUIRE(j.contains("routing"));
    REQUIRE(j.contains("metrics"));

    std::filesystem::remove_all(tmp);
}

TEST_CASE("LaTeX output contains table environment and booktabs commands", "[benchmark]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_test_latex";
    std::filesystem::create_directories(tmp);
    std::string tex_path = (tmp / "tables.tex").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    populate_test_metrics(metrics);
    reporter.write_latex(tex_path, metrics, make_test_vmm_stats(), make_test_routing());

    auto content = slurp(tex_path);
    REQUIRE_THAT(content, ContainsSubstring("\\begin{table}"));
    REQUIRE_THAT(content, ContainsSubstring("\\end{table}"));
    REQUIRE_THAT(content, ContainsSubstring("\\toprule"));
    REQUIRE_THAT(content, ContainsSubstring("\\midrule"));
    REQUIRE_THAT(content, ContainsSubstring("\\bottomrule"));
    // Three tables
    REQUIRE_THAT(content, ContainsSubstring("Performance Summary"));
    REQUIRE_THAT(content, ContainsSubstring("Routing Analysis"));
    REQUIRE_THAT(content, ContainsSubstring("Memory Efficiency"));

    std::filesystem::remove_all(tmp);
}

TEST_CASE("LaTeX standalone mode includes documentclass", "[benchmark]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_test_standalone";
    std::filesystem::create_directories(tmp);
    std::string tex_path = (tmp / "tables.tex").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, true});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    populate_test_metrics(metrics);
    reporter.write_latex(tex_path, metrics, make_test_vmm_stats(), make_test_routing());

    auto content = slurp(tex_path);
    REQUIRE_THAT(content, ContainsSubstring("\\documentclass{article}"));
    REQUIRE_THAT(content, ContainsSubstring("\\usepackage{booktabs}"));
    REQUIRE_THAT(content, ContainsSubstring("\\begin{document}"));
    REQUIRE_THAT(content, ContainsSubstring("\\end{document}"));

    std::filesystem::remove_all(tmp);
}

TEST_CASE("LaTeX non-standalone does NOT include documentclass", "[benchmark]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_test_nostandalone";
    std::filesystem::create_directories(tmp);
    std::string tex_path = (tmp / "tables.tex").string();

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("test_model", 100, 5000.0, 50.0);

    nos::MetricsCollector metrics;
    populate_test_metrics(metrics);
    reporter.write_latex(tex_path, metrics, make_test_vmm_stats(), make_test_routing());

    auto content = slurp(tex_path);
    REQUIRE_FALSE(content.find("\\documentclass") != std::string::npos);
    REQUIRE_FALSE(content.find("\\begin{document}") != std::string::npos);

    std::filesystem::remove_all(tmp);
}

TEST_CASE("escape_latex handles special characters", "[benchmark]") {
    REQUIRE(nos::BenchmarkReporter::escape_latex("a&b") == "a\\&b");
    REQUIRE(nos::BenchmarkReporter::escape_latex("100%") == "100\\%");
    REQUIRE(nos::BenchmarkReporter::escape_latex("$x$") == "\\$x\\$");
    REQUIRE(nos::BenchmarkReporter::escape_latex("#1") == "\\#1");
    REQUIRE(nos::BenchmarkReporter::escape_latex("a_b") == "a\\_b");
    REQUIRE(nos::BenchmarkReporter::escape_latex("{x}") == "\\{x\\}");
    REQUIRE(nos::BenchmarkReporter::escape_latex("~^") == "\\textasciitilde{}\\textasciicircum{}");
}

TEST_CASE("write creates output directory if missing", "[benchmark]") {
    auto tmp = std::filesystem::temp_directory_path() / "nos_bench_test_mkdir" / "nested";
    if (std::filesystem::exists(tmp)) std::filesystem::remove_all(tmp);

    nos::BenchmarkReporter reporter({{tmp.string()}, false});
    reporter.set_run_info("model", 10, 1000.0, 10.0);

    nos::MetricsCollector metrics;
    populate_test_metrics(metrics);
    reporter.write(metrics, make_test_vmm_stats(), make_test_routing());

    REQUIRE(std::filesystem::exists(tmp / "benchmark_results.csv"));
    REQUIRE(std::filesystem::exists(tmp / "benchmark_results.json"));
    REQUIRE(std::filesystem::exists(tmp / "benchmark_tables.tex"));

    std::filesystem::remove_all(tmp.parent_path());
}
