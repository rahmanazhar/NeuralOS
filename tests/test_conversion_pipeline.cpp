/// @file test_conversion_pipeline.cpp
/// @brief Tests for the end-to-end conversion pipeline.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "converter/conversion_pipeline.h"
#include "converter/model_config.h"
#include "format/expert_format.h"
#include "kernel/packing.h"

namespace fs = std::filesystem;
using Catch::Matchers::WithinAbs;

/// Helper: Create a synthetic tiny model as SafeTensors format files.
///
/// Model: 2 layers, hidden_dim=64, intermediate_dim=128, vocab_size=256, 4 heads
/// Creates: config.json + model.safetensors with embedding, per-layer FFN and attention weights.
static std::string create_synthetic_model(const std::string& base_dir) {
    std::string model_dir = base_dir + "/tiny_model";
    fs::create_directories(model_dir);

    // Write config.json
    nlohmann::json config = {
        {"model_type", "llama"},
        {"vocab_size", 256},
        {"hidden_size", 64},
        {"intermediate_size", 128},
        {"num_hidden_layers", 2},
        {"num_attention_heads", 4},
        {"num_key_value_heads", 4},
        {"max_position_embeddings", 512},
        {"rope_theta", 10000.0},
        {"rms_norm_eps", 1e-5}
    };
    {
        std::ofstream ofs(model_dir + "/config.json");
        ofs << config.dump(2);
    }

    // Build SafeTensors data
    // We need:
    // - model.embed_tokens.weight [256 x 64] F32
    // - Per layer (0, 1):
    //   - model.layers.N.mlp.gate_proj.weight [128 x 64] F32
    //   - model.layers.N.mlp.up_proj.weight [128 x 64] F32
    //   - model.layers.N.mlp.down_proj.weight [64 x 128] F32
    //   - model.layers.N.self_attn.q_proj.weight [64 x 64] F32
    //   - model.layers.N.self_attn.k_proj.weight [64 x 64] F32
    //   - model.layers.N.self_attn.v_proj.weight [64 x 64] F32
    //   - model.layers.N.self_attn.o_proj.weight [64 x 64] F32

    struct TensorSpec {
        std::string name;
        int rows, cols;
    };

    std::vector<TensorSpec> specs;
    specs.push_back({"model.embed_tokens.weight", 256, 64});

    for (int layer = 0; layer < 2; layer++) {
        std::string prefix = "model.layers." + std::to_string(layer);
        specs.push_back({prefix + ".mlp.gate_proj.weight", 128, 64});
        specs.push_back({prefix + ".mlp.up_proj.weight", 128, 64});
        specs.push_back({prefix + ".mlp.down_proj.weight", 64, 128});
        specs.push_back({prefix + ".self_attn.q_proj.weight", 64, 64});
        specs.push_back({prefix + ".self_attn.k_proj.weight", 64, 64});
        specs.push_back({prefix + ".self_attn.v_proj.weight", 64, 64});
        specs.push_back({prefix + ".self_attn.o_proj.weight", 64, 64});
    }

    // Calculate total data size and build header
    nlohmann::json header;
    uint64_t offset = 0;

    for (const auto& spec : specs) {
        uint64_t size = static_cast<uint64_t>(spec.rows) * static_cast<uint64_t>(spec.cols) * 4;
        header[spec.name] = {
            {"dtype", "F32"},
            {"shape", {spec.rows, spec.cols}},
            {"data_offsets", {offset, offset + size}}
        };
        offset += size;
    }

    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();

    // Write SafeTensors file
    std::string st_path = model_dir + "/model.safetensors";
    std::ofstream ofs(st_path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&header_size), 8);
    ofs.write(header_str.data(), static_cast<std::streamsize>(header_size));

    // Generate deterministic random weights
    uint64_t rng_state = 12345;
    auto next_float = [&rng_state]() -> float {
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>(rng_state >> 33) / static_cast<float>(1ULL << 31)) - 1.0f;
    };

    for (const auto& spec : specs) {
        int elems = spec.rows * spec.cols;
        for (int i = 0; i < elems; i++) {
            float val = next_float() * 0.1f;  // small random values
            ofs.write(reinterpret_cast<const char*>(&val), 4);
        }
    }

    return model_dir;
}

TEST_CASE("ConversionPipeline: full conversion of synthetic model", "[conversion]") {
    auto tmpdir = fs::temp_directory_path() / "test_conv_pipeline";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);

    std::string model_dir = create_synthetic_model(tmpdir.string());
    std::string output_dir = tmpdir.string() + "/output";

    nos::ConversionConfig conv_cfg;
    conv_cfg.input_path = model_dir;
    conv_cfg.output_dir = output_dir;
    conv_cfg.target_expert_size_mb = 1;  // Small target -> more experts per layer
    conv_cfg.calibration_samples = 32;
    conv_cfg.top_k = 2;
    conv_cfg.resume = true;
    conv_cfg.calibration_data_path = "";  // Synthetic mode

    nos::ConversionPipeline pipeline;
    REQUIRE(pipeline.run(conv_cfg));

    SECTION("NXP output file exists and is valid") {
        std::string nxp_path = output_dir + "/model.nxp";
        REQUIRE(fs::exists(nxp_path));

        nos::NxpReader reader;
        REQUIRE(reader.open(nxp_path));

        CHECK(reader.header().magic == nos::NXP_MAGIC);
        CHECK(reader.header().version == nos::NXP_VERSION);
        CHECK(reader.header().num_layers == 2);
        CHECK(reader.header().hidden_dim == 64);
        CHECK(reader.header().intermediate_dim == 128);
        CHECK(reader.num_entries() > 0);
    }

    SECTION("model_config.json round-trip") {
        std::string config_path = output_dir + "/model_config.json";
        REQUIRE(fs::exists(config_path));

        std::ifstream ifs(config_path);
        nlohmann::json j = nlohmann::json::parse(ifs);
        nos::ModelConfig mc = j.get<nos::ModelConfig>();

        CHECK(mc.architecture == "llama");
        CHECK(mc.vocab_size == 256);
        CHECK(mc.hidden_dim == 64);
        CHECK(mc.intermediate_dim == 128);
        CHECK(mc.n_layers == 2);
        CHECK(mc.n_heads == 4);
        CHECK(mc.n_kv_heads == 4);
        CHECK(mc.head_dim == 16);
        CHECK(mc.max_seq_len == 512);
        CHECK(mc.expert_count >= 2);  // At least 2 experts
        CHECK(mc.top_k == 2);
        CHECK(mc.attention_type == "mha");
        CHECK_THAT(static_cast<double>(mc.rope_theta), WithinAbs(10000.0, 1.0));
        CHECK_THAT(static_cast<double>(mc.norm_eps), WithinAbs(1e-5, 1e-10));

        // All 14 fields present
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
    }

    SECTION("checkpoint removed after completion") {
        std::string checkpoint_path = output_dir + "/.checkpoint.json";
        CHECK_FALSE(fs::exists(checkpoint_path));
    }

    fs::remove_all(tmpdir);
}

TEST_CASE("ConversionPipeline: checkpoint-based resume", "[conversion]") {
    auto tmpdir = fs::temp_directory_path() / "test_conv_resume";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);

    std::string model_dir = create_synthetic_model(tmpdir.string());
    std::string output_dir = tmpdir.string() + "/output_resume";
    fs::create_directories(output_dir);

    // Create a fake checkpoint at layer 1 (first layer complete)
    {
        nlohmann::json cp = {
            {"completed_layers", 1},
            {"nxp_offset", 0}
        };
        std::ofstream ofs(output_dir + "/.checkpoint.json");
        ofs << cp.dump(2);
    }

    nos::ConversionConfig conv_cfg;
    conv_cfg.input_path = model_dir;
    conv_cfg.output_dir = output_dir;
    conv_cfg.target_expert_size_mb = 1;
    conv_cfg.calibration_samples = 16;
    conv_cfg.top_k = 2;
    conv_cfg.resume = true;
    conv_cfg.calibration_data_path = "";

    nos::ConversionPipeline pipeline;

    // This should succeed -- the pipeline restarts because the nxp file doesn't exist yet
    // but the checkpoint logic is exercised
    REQUIRE(pipeline.run(conv_cfg));

    // Verify output is valid
    std::string config_path = output_dir + "/model_config.json";
    REQUIRE(fs::exists(config_path));

    std::ifstream ifs(config_path);
    nlohmann::json j = nlohmann::json::parse(ifs);
    nos::ModelConfig mc = j.get<nos::ModelConfig>();
    CHECK(mc.n_layers == 2);
    CHECK(mc.expert_count >= 2);

    // Checkpoint should be removed
    CHECK_FALSE(fs::exists(output_dir + "/.checkpoint.json"));

    fs::remove_all(tmpdir);
}

TEST_CASE("ConversionPipeline: invalid input path fails gracefully", "[conversion]") {
    nos::ConversionConfig conv_cfg;
    conv_cfg.input_path = "/nonexistent/model";
    conv_cfg.output_dir = "/tmp/test_conv_fail";
    conv_cfg.calibration_data_path = "";

    nos::ConversionPipeline pipeline;
    CHECK_FALSE(pipeline.run(conv_cfg));
}
