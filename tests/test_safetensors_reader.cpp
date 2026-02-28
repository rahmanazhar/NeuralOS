/// @file test_safetensors_reader.cpp
/// @brief Tests for SafeTensors format reader.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "converter/safetensors_reader.h"

namespace fs = std::filesystem;
using Catch::Matchers::WithinAbs;

/// Helper: write a minimal safetensors file with one F32 tensor [rows x cols].
static std::string write_safetensors(const std::string& dir,
                                      const std::string& filename,
                                      const std::string& tensor_name,
                                      int rows, int cols,
                                      const float* data) {
    std::string path = dir + "/" + filename;

    // Build header JSON
    nlohmann::json header;
    uint64_t data_size = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * 4;
    header[tensor_name] = {
        {"dtype", "F32"},
        {"shape", {rows, cols}},
        {"data_offsets", {0, data_size}}
    };

    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();

    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&header_size), 8);
    ofs.write(header_str.data(), static_cast<std::streamsize>(header_size));
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(data_size));
    ofs.close();

    return path;
}

/// Helper: write a config.json for a small model.
static void write_config_json(const std::string& dir,
                               uint32_t vocab = 256,
                               uint32_t hidden = 64,
                               uint32_t intermediate = 128,
                               uint32_t layers = 2,
                               uint32_t heads = 4,
                               uint32_t kv_heads = 4) {
    nlohmann::json j = {
        {"model_type", "llama"},
        {"vocab_size", vocab},
        {"hidden_size", hidden},
        {"intermediate_size", intermediate},
        {"num_hidden_layers", layers},
        {"num_attention_heads", heads},
        {"num_key_value_heads", kv_heads},
        {"max_position_embeddings", 2048},
        {"rope_theta", 10000.0},
        {"rms_norm_eps", 1e-5}
    };

    std::ofstream ofs(dir + "/config.json");
    ofs << j.dump(2);
}

TEST_CASE("SafeTensors: parse single-shard file", "[safetensors]") {
    auto tmpdir = fs::temp_directory_path() / "test_st_parse";
    fs::create_directories(tmpdir);

    // Create test data: 4x8 float matrix
    std::vector<float> data(32);
    for (int i = 0; i < 32; i++) data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f;

    write_safetensors(tmpdir.string(), "model.safetensors", "test.weight", 4, 8, data.data());
    write_config_json(tmpdir.string());

    nos::SafeTensorsReader reader;
    REQUIRE(reader.open(tmpdir.string()));

    SECTION("tensor names") {
        auto names = reader.tensor_names();
        REQUIRE(names.size() == 1);
        REQUIRE(names[0] == "test.weight");
    }

    SECTION("find tensor") {
        auto* info = reader.find_tensor("test.weight");
        REQUIRE(info != nullptr);
        CHECK(info->dtype == "F32");
        CHECK(info->shape.size() == 2);
        CHECK(info->shape[0] == 4);
        CHECK(info->shape[1] == 8);
        CHECK(info->numel() == 32);
        CHECK(info->element_size() == 4);
    }

    SECTION("read full tensor") {
        auto* info = reader.find_tensor("test.weight");
        REQUIRE(info != nullptr);

        std::vector<float> buf(32);
        REQUIRE(reader.read_tensor(*info, buf.data(), buf.size() * sizeof(float)));

        for (int i = 0; i < 32; i++) {
            CHECK_THAT(buf[static_cast<size_t>(i)], WithinAbs(static_cast<float>(i) * 0.1f, 1e-6f));
        }
    }

    SECTION("read tensor rows") {
        auto* info = reader.find_tensor("test.weight");
        REQUIRE(info != nullptr);

        // Read rows 1-2 (indices 1 and 2)
        std::vector<float> buf(16);
        REQUIRE(reader.read_tensor_rows(*info, 1, 2, buf.data(), buf.size() * sizeof(float)));

        // Row 1 starts at index 8
        for (int i = 0; i < 16; i++) {
            float expected = static_cast<float>(i + 8) * 0.1f;
            CHECK_THAT(buf[static_cast<size_t>(i)], WithinAbs(expected, 1e-6f));
        }
    }

    SECTION("config extraction") {
        auto cfg = reader.config();
        CHECK(cfg.architecture == "llama");
        CHECK(cfg.vocab_size == 256);
        CHECK(cfg.hidden_dim == 64);
        CHECK(cfg.intermediate_dim == 128);
        CHECK(cfg.n_layers == 2);
        CHECK(cfg.n_heads == 4);
        CHECK(cfg.n_kv_heads == 4);
        CHECK(cfg.head_dim == 16);
        CHECK(cfg.attention_type == "mha");
    }

    SECTION("missing tensor returns nullptr") {
        CHECK(reader.find_tensor("nonexistent") == nullptr);
    }

    fs::remove_all(tmpdir);
}

TEST_CASE("SafeTensors: F16 tensor dtype", "[safetensors]") {
    auto tmpdir = fs::temp_directory_path() / "test_st_f16";
    fs::create_directories(tmpdir);

    // Write a safetensors file with F16 tensor header (but use dummy data)
    nlohmann::json header;
    uint64_t data_size = 4 * 8 * 2;  // 4x8 F16
    header["emb.weight"] = {
        {"dtype", "F16"},
        {"shape", {4, 8}},
        {"data_offsets", {0, data_size}}
    };
    std::string header_str = header.dump();
    uint64_t header_len = header_str.size();

    std::string path = tmpdir.string() + "/model.safetensors";
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&header_len), 8);
    ofs.write(header_str.data(), static_cast<std::streamsize>(header_len));
    std::vector<uint16_t> dummy_f16(32, 0);
    ofs.write(reinterpret_cast<const char*>(dummy_f16.data()), static_cast<std::streamsize>(data_size));
    ofs.close();

    write_config_json(tmpdir.string());

    nos::SafeTensorsReader reader;
    REQUIRE(reader.open(tmpdir.string()));

    auto* info = reader.find_tensor("emb.weight");
    REQUIRE(info != nullptr);
    CHECK(info->dtype == "F16");
    CHECK(info->element_size() == 2);

    fs::remove_all(tmpdir);
}

TEST_CASE("SafeTensors: error cases", "[safetensors]") {
    SECTION("nonexistent path") {
        nos::SafeTensorsReader reader;
        CHECK_FALSE(reader.open("/nonexistent/path"));
    }

    SECTION("truncated header") {
        auto tmpdir = fs::temp_directory_path() / "test_st_trunc";
        fs::create_directories(tmpdir);

        // Write only 4 bytes (less than 8-byte header size field)
        std::string path = tmpdir.string() + "/model.safetensors";
        std::ofstream ofs(path, std::ios::binary);
        uint32_t junk = 42;
        ofs.write(reinterpret_cast<const char*>(&junk), 4);
        ofs.close();

        write_config_json(tmpdir.string());

        nos::SafeTensorsReader reader;
        CHECK_FALSE(reader.open(tmpdir.string()));

        fs::remove_all(tmpdir);
    }

    SECTION("missing config.json") {
        auto tmpdir = fs::temp_directory_path() / "test_st_nocfg";
        fs::create_directories(tmpdir);

        std::vector<float> data(8, 1.0f);
        write_safetensors(tmpdir.string(), "model.safetensors", "w", 2, 4, data.data());
        // No config.json

        nos::SafeTensorsReader reader;
        CHECK_FALSE(reader.open(tmpdir.string()));

        fs::remove_all(tmpdir);
    }
}

TEST_CASE("SafeTensors: GQA config (n_kv_heads < n_heads)", "[safetensors]") {
    auto tmpdir = fs::temp_directory_path() / "test_st_gqa";
    fs::create_directories(tmpdir);

    std::vector<float> data(8, 0.0f);
    write_safetensors(tmpdir.string(), "model.safetensors", "w", 2, 4, data.data());
    write_config_json(tmpdir.string(), 32000, 4096, 11008, 32, 32, 8);

    nos::SafeTensorsReader reader;
    REQUIRE(reader.open(tmpdir.string()));

    auto cfg = reader.config();
    CHECK(cfg.n_heads == 32);
    CHECK(cfg.n_kv_heads == 8);
    CHECK(cfg.attention_type == "gqa");
    CHECK(cfg.head_dim == 128);

    fs::remove_all(tmpdir);
}
