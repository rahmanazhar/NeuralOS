/// @file test_model_config.cpp
/// @brief Tests for ModelConfig JSON serialization and schema.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "converter/model_config.h"

using Catch::Matchers::WithinAbs;
namespace fs = std::filesystem;

TEST_CASE("ModelConfig: JSON round-trip with all 14 fields", "[model_config]") {
    nos::ModelConfig original;
    original.architecture = "llama";
    original.vocab_size = 32000;
    original.hidden_dim = 4096;
    original.intermediate_dim = 11008;
    original.n_layers = 32;
    original.n_heads = 32;
    original.n_kv_heads = 32;
    original.head_dim = 128;
    original.max_seq_len = 4096;
    original.rope_theta = 10000.0f;
    original.norm_eps = 1e-5f;
    original.expert_count = 8;
    original.top_k = 2;
    original.max_expert_size = 104857600;
    original.attention_type = "mha";

    // Serialize
    nlohmann::json j = original;

    // Verify all 14 fields present
    CHECK(j.contains("architecture"));
    CHECK(j.contains("vocab_size"));
    CHECK(j.contains("hidden_dim"));
    CHECK(j.contains("intermediate_dim"));
    CHECK(j.contains("n_layers"));
    CHECK(j.contains("n_heads"));
    CHECK(j.contains("n_kv_heads"));
    CHECK(j.contains("head_dim"));
    CHECK(j.contains("max_seq_len"));
    CHECK(j.contains("rope_theta"));
    CHECK(j.contains("norm_eps"));
    CHECK(j.contains("expert_count"));
    CHECK(j.contains("top_k"));
    CHECK(j.contains("max_expert_size"));
    CHECK(j.contains("attention_type"));

    // Deserialize
    nos::ModelConfig restored = j.get<nos::ModelConfig>();

    // Verify all fields match
    CHECK(restored.architecture == original.architecture);
    CHECK(restored.vocab_size == original.vocab_size);
    CHECK(restored.hidden_dim == original.hidden_dim);
    CHECK(restored.intermediate_dim == original.intermediate_dim);
    CHECK(restored.n_layers == original.n_layers);
    CHECK(restored.n_heads == original.n_heads);
    CHECK(restored.n_kv_heads == original.n_kv_heads);
    CHECK(restored.head_dim == original.head_dim);
    CHECK(restored.max_seq_len == original.max_seq_len);
    CHECK_THAT(static_cast<double>(restored.rope_theta), WithinAbs(10000.0, 1e-3));
    CHECK_THAT(static_cast<double>(restored.norm_eps), WithinAbs(1e-5, 1e-10));
    CHECK(restored.expert_count == original.expert_count);
    CHECK(restored.top_k == original.top_k);
    CHECK(restored.max_expert_size == original.max_expert_size);
    CHECK(restored.attention_type == original.attention_type);
}

TEST_CASE("ModelConfig: Llama-2-7B config", "[model_config]") {
    nos::ModelConfig cfg;
    cfg.architecture = "llama";
    cfg.vocab_size = 32000;
    cfg.hidden_dim = 4096;
    cfg.intermediate_dim = 11008;
    cfg.n_layers = 32;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 32;
    cfg.head_dim = 128;
    cfg.max_seq_len = 4096;

    nlohmann::json j = cfg;
    nos::ModelConfig restored = j.get<nos::ModelConfig>();

    CHECK(restored.vocab_size == 32000);
    CHECK(restored.hidden_dim == 4096);
    CHECK(restored.intermediate_dim == 11008);
    CHECK(restored.n_layers == 32);
}

TEST_CASE("ModelConfig: GQA config", "[model_config]") {
    nos::ModelConfig cfg;
    cfg.n_heads = 32;
    cfg.n_kv_heads = 8;
    cfg.attention_type = "gqa";

    nlohmann::json j = cfg;
    nos::ModelConfig restored = j.get<nos::ModelConfig>();

    CHECK(restored.n_kv_heads == 8);
    CHECK(restored.attention_type == "gqa");
}

TEST_CASE("ModelConfig: derive_attention_type", "[model_config]") {
    CHECK(nos::derive_attention_type(32, 32) == "mha");
    CHECK(nos::derive_attention_type(32, 8) == "gqa");
    CHECK(nos::derive_attention_type(32, 1) == "mqa");
}

TEST_CASE("ModelConfig: missing JSON field throws", "[model_config]") {
    // Create JSON with a missing required field
    nlohmann::json j = {
        {"architecture", "llama"},
        {"vocab_size", 32000},
        // missing "hidden_dim" and others
    };

    CHECK_THROWS_AS(j.get<nos::ModelConfig>(), nlohmann::json::out_of_range);
}

TEST_CASE("ModelConfig: file write and read back", "[model_config]") {
    auto tmpdir = fs::temp_directory_path() / "test_mc_file";
    fs::create_directories(tmpdir);
    std::string path = tmpdir.string() + "/model_config.json";

    nos::ModelConfig original;
    original.architecture = "llama";
    original.vocab_size = 32000;
    original.hidden_dim = 4096;
    original.intermediate_dim = 11008;
    original.n_layers = 32;
    original.n_heads = 32;
    original.n_kv_heads = 8;
    original.head_dim = 128;
    original.max_seq_len = 4096;
    original.rope_theta = 500000.0f;
    original.norm_eps = 1e-5f;
    original.expert_count = 4;
    original.top_k = 2;
    original.max_expert_size = 52428800;
    original.attention_type = "gqa";

    // Write
    {
        nlohmann::json j = original;
        std::ofstream ofs(path);
        ofs << j.dump(2);
    }

    // Read back
    {
        std::ifstream ifs(path);
        nlohmann::json j = nlohmann::json::parse(ifs);
        nos::ModelConfig restored = j.get<nos::ModelConfig>();

        CHECK(restored.architecture == "llama");
        CHECK(restored.vocab_size == 32000);
        CHECK(restored.hidden_dim == 4096);
        CHECK(restored.n_kv_heads == 8);
        CHECK(restored.expert_count == 4);
        CHECK(restored.attention_type == "gqa");
        CHECK_THAT(static_cast<double>(restored.rope_theta), WithinAbs(500000.0, 1.0));
    }

    fs::remove_all(tmpdir);
}
