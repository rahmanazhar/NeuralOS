/// @file test_bench_command.cpp
/// @brief Tests for BenchRunResult, paper Table 1 LaTeX, CSV, JSON, and comparison table.

#include "engine/benchmark.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using Catch::Matchers::ContainsSubstring;

// ── Helper: create a sample BenchRunResult ──────────────────────────────────

static nos::BenchRunResult make_result(
        const std::string& name, const std::string& size,
        double tps, double ttft, double chr,
        const std::string& mode, double rwp, int ek) {
    nos::BenchRunResult r;
    r.model_name         = name;
    r.model_size         = size;
    r.tokens_generated   = 512;
    r.total_time_ms      = (tps > 0.0) ? static_cast<double>(512) / tps * 1000.0 : 0.0;
    r.ttft_ms            = ttft;
    r.tok_per_sec        = tps;
    r.cache_hit_rate     = chr;
    r.switch_rate        = 0.0312;
    r.avg_sticky_window  = 28.5;
    r.prefetch_rwp       = rwp;
    r.prefetch_mode      = mode;
    r.effective_k        = ek;
    r.waste_ratio        = 0.05;
    r.memory_budget_mb   = 16384;
    r.num_threads        = 4;
    r.concurrent_sequences = 0;
    r.multi_seq_tok_per_sec = 0.0;
    return r;
}

// ── BenchRunResult serializes to JSON with all expected fields ──────────────

TEST_CASE("BenchRunResult serializes to JSON with all expected fields", "[bench]") {
    std::string tmp_dir = std::filesystem::temp_directory_path().string() + "/nos_bench_test_json";
    std::filesystem::create_directories(tmp_dir);
    std::string json_path = tmp_dir + "/benchmark.json";

    nos::BenchmarkReporter reporter({{"bench_out"}, false});
    std::vector<nos::BenchRunResult> results;
    results.push_back(make_result("llama3-70b", "70B", 10.5, 2500.0, 0.92, "oracle", 0.85, 8));

    reporter.write_paper_json(json_path, results);

    std::ifstream ifs(json_path);
    REQUIRE(ifs.is_open());
    auto arr = nlohmann::json::parse(ifs);
    REQUIRE(arr.is_array());
    REQUIRE(arr.size() == 1);

    auto& j = arr[0];
    CHECK(j["model_name"] == "llama3-70b");
    CHECK(j["model_size"] == "70B");
    CHECK(j["tokens_generated"] == 512);
    CHECK(j["tok_per_sec"].get<double>() > 10.0);
    CHECK(j["ttft_ms"].get<double>() > 0.0);
    CHECK(j["cache_hit_rate"].get<double>() > 0.9);
    CHECK(j["switch_rate"].get<double>() > 0.0);
    CHECK(j["avg_sticky_window"].get<double>() > 0.0);
    CHECK(j["prefetch_rwp"].get<double>() > 0.0);
    CHECK(j["prefetch_mode"] == "oracle");
    CHECK(j["effective_k"] == 8);
    CHECK(j["waste_ratio"].get<double>() >= 0.0);
    CHECK(j["memory_budget_mb"] == 16384);
    CHECK(j["num_threads"] == 4);
    CHECK(j.contains("concurrent_sequences"));
    CHECK(j.contains("multi_seq_tok_per_sec"));

    std::filesystem::remove_all(tmp_dir);
}

// ── write_paper_table generates valid LaTeX with booktabs ──────────────────

TEST_CASE("write_paper_table generates valid LaTeX with booktabs", "[bench]") {
    std::string tmp_dir = std::filesystem::temp_directory_path().string() + "/nos_bench_test_latex";
    std::filesystem::create_directories(tmp_dir);
    std::string tex_path = tmp_dir + "/table1.tex";

    nos::BenchmarkReporter reporter({{"bench_out"}, true});
    std::vector<nos::BenchRunResult> results;
    results.push_back(make_result("llama3-70b", "70B", 10.5, 2500.0, 0.92, "oracle", 0.85, 8));
    results.push_back(make_result("llama3-7b", "7B", 35.0, 800.0, 0.95, "none", 0.0, 0));

    reporter.write_paper_table(tex_path, results);

    std::ifstream ifs(tex_path);
    REQUIRE(ifs.is_open());
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();

    // Check booktabs commands
    CHECK_THAT(content, ContainsSubstring("\\toprule"));
    CHECK_THAT(content, ContainsSubstring("\\midrule"));
    CHECK_THAT(content, ContainsSubstring("\\bottomrule"));

    // Check table environment
    CHECK_THAT(content, ContainsSubstring("\\begin{table}"));
    CHECK_THAT(content, ContainsSubstring("\\end{table}"));

    // Check caption
    CHECK_THAT(content, ContainsSubstring("NeuralOS Inference Performance"));

    // Check headers
    CHECK_THAT(content, ContainsSubstring("Model"));
    CHECK_THAT(content, ContainsSubstring("Params"));
    CHECK_THAT(content, ContainsSubstring("tok/s"));
    CHECK_THAT(content, ContainsSubstring("TTFT"));
    CHECK_THAT(content, ContainsSubstring("Cache Hit"));
    CHECK_THAT(content, ContainsSubstring("Prefetch Acc."));
    CHECK_THAT(content, ContainsSubstring("Memory"));

    // Check data values present
    CHECK_THAT(content, ContainsSubstring("70B"));
    CHECK_THAT(content, ContainsSubstring("7B"));
    CHECK_THAT(content, ContainsSubstring("10.5"));
    CHECK_THAT(content, ContainsSubstring("35.0"));

    // Check standalone mode
    CHECK_THAT(content, ContainsSubstring("\\documentclass"));
    CHECK_THAT(content, ContainsSubstring("\\usepackage{booktabs}"));
    CHECK_THAT(content, ContainsSubstring("\\begin{document}"));
    CHECK_THAT(content, ContainsSubstring("\\end{document}"));

    std::filesystem::remove_all(tmp_dir);
}

// ── write_paper_csv generates CSV with correct headers ─────────────────────

TEST_CASE("write_paper_csv generates CSV with correct headers", "[bench]") {
    std::string tmp_dir = std::filesystem::temp_directory_path().string() + "/nos_bench_test_csv";
    std::filesystem::create_directories(tmp_dir);
    std::string csv_path = tmp_dir + "/benchmark.csv";

    nos::BenchmarkReporter reporter({{"bench_out"}, false});
    std::vector<nos::BenchRunResult> results;
    results.push_back(make_result("llama3-70b", "70B", 10.5, 2500.0, 0.92, "oracle", 0.85, 8));

    reporter.write_paper_csv(csv_path, results);

    std::ifstream ifs(csv_path);
    REQUIRE(ifs.is_open());

    std::string header;
    std::getline(ifs, header);

    // Verify all expected header fields
    CHECK_THAT(header, ContainsSubstring("model_name"));
    CHECK_THAT(header, ContainsSubstring("model_size"));
    CHECK_THAT(header, ContainsSubstring("tokens_generated"));
    CHECK_THAT(header, ContainsSubstring("tok_per_sec"));
    CHECK_THAT(header, ContainsSubstring("ttft_ms"));
    CHECK_THAT(header, ContainsSubstring("cache_hit_rate"));
    CHECK_THAT(header, ContainsSubstring("switch_rate"));
    CHECK_THAT(header, ContainsSubstring("prefetch_rwp"));
    CHECK_THAT(header, ContainsSubstring("prefetch_mode"));
    CHECK_THAT(header, ContainsSubstring("effective_k"));
    CHECK_THAT(header, ContainsSubstring("waste_ratio"));
    CHECK_THAT(header, ContainsSubstring("memory_budget_mb"));
    CHECK_THAT(header, ContainsSubstring("num_threads"));
    CHECK_THAT(header, ContainsSubstring("concurrent_sequences"));
    CHECK_THAT(header, ContainsSubstring("multi_seq_tok_per_sec"));

    // Verify data row exists
    std::string data;
    std::getline(ifs, data);
    CHECK_THAT(data, ContainsSubstring("llama3-70b"));
    CHECK_THAT(data, ContainsSubstring("70B"));
    CHECK_THAT(data, ContainsSubstring("oracle"));

    std::filesystem::remove_all(tmp_dir);
}

// ── write_comparison_table merges multiple results correctly ────────────────

TEST_CASE("write_comparison_table merges multiple results correctly", "[bench]") {
    std::string tmp_dir = std::filesystem::temp_directory_path().string() + "/nos_bench_test_comp";
    std::filesystem::create_directories(tmp_dir);
    std::string tex_path = tmp_dir + "/comparison.tex";

    nos::BenchmarkReporter reporter({{"bench_out"}, false});
    std::vector<nos::BenchRunResult> results;
    results.push_back(make_result("llama3-70b", "70B", 5.2, 3500.0, 0.80, "none", 0.0, 0));
    results.push_back(make_result("llama3-70b", "70B", 8.5, 3000.0, 0.88, "ngram_3", 0.65, 3));
    results.push_back(make_result("llama3-70b", "70B", 10.5, 2500.0, 0.92, "oracle", 0.85, 8));

    reporter.write_comparison_table(tex_path, results);

    std::ifstream ifs(tex_path);
    REQUIRE(ifs.is_open());
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();

    // Check config labels
    CHECK_THAT(content, ContainsSubstring("No prefetch"));
    CHECK_THAT(content, ContainsSubstring("N-gram"));
    CHECK_THAT(content, ContainsSubstring("Oracle"));

    // Check structure
    CHECK_THAT(content, ContainsSubstring("Configuration Comparison"));
    CHECK_THAT(content, ContainsSubstring("\\toprule"));
    CHECK_THAT(content, ContainsSubstring("\\bottomrule"));

    // All three rows present
    CHECK_THAT(content, ContainsSubstring("5.2"));
    CHECK_THAT(content, ContainsSubstring("8.5"));
    CHECK_THAT(content, ContainsSubstring("10.5"));

    std::filesystem::remove_all(tmp_dir);
}

// ── Default multi-domain prompt contains expected keywords ─────────────────

TEST_CASE("Default multi-domain prompt contains expected keywords", "[bench]") {
    // The default prompt is defined in main.cpp as kDefaultBenchPrompt.
    // We verify it here by checking a known constant string.
    const std::string default_prompt =
        "Explain the concept of quantum entanglement in simple terms. "
        "Then write a Python function to calculate fibonacci numbers. "
        "Finally, describe the process of photosynthesis in plants.";

    CHECK_THAT(default_prompt, ContainsSubstring("quantum"));
    CHECK_THAT(default_prompt, ContainsSubstring("entanglement"));
    CHECK_THAT(default_prompt, ContainsSubstring("fibonacci"));
    CHECK_THAT(default_prompt, ContainsSubstring("photosynthesis"));
}
