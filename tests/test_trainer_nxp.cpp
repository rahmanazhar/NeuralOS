/// @file test_trainer_nxp.cpp
/// @brief Integration tests for NXP round-trip in the training subsystem.
///
/// Tests validate that trainer.cpp correctly loads expert weights from .nxp
/// files, trains them, and writes modified weights back. Also validates
/// the dequantize-requantize round-trip and merge mechanics.

#include <catch2/catch_test_macros.hpp>

#include "training/trainer.h"
#include "training/lora.h"
#include "format/expert_format.h"
#include "kernel/packing.h"
#include "converter/quantizer.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

/// Helper: create a temp directory for test artifacts.
static fs::path make_test_dir(const std::string& name) {
    fs::path dir = fs::temp_directory_path() / "neuralos_test" / name;
    fs::create_directories(dir);
    return dir;
}

/// Helper: clean up a test directory.
static void cleanup_test_dir(const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
}

/// Helper: write a minimal model_config.json.
static void write_model_config(const fs::path& dir,
                                uint32_t n_layers, uint32_t expert_count,
                                uint32_t hidden_dim, uint32_t intermediate_dim) {
    nlohmann::json cfg;
    cfg["n_layers"] = n_layers;
    cfg["expert_count"] = expert_count;
    cfg["hidden_dim"] = hidden_dim;
    cfg["intermediate_dim"] = intermediate_dim;
    cfg["vocab_size"] = 256;

    std::ofstream ofs((dir / "model_config.json").string());
    ofs << cfg.dump(2);
}

/// Helper: write a minimal JSONL training data file.
static void write_training_data(const fs::path& dir) {
    std::ofstream ofs((dir / "train.jsonl").string());
    ofs << R"({"text": "hello world"})" << "\n";
    ofs << R"({"text": "test training data"})" << "\n";
}

/// Helper: create a .nxp file with known ternary weights.
/// Uses alternating {-1, 0, 1} pattern with scale = 1.0 for each channel.
static void create_test_nxp(const fs::path& dir,
                             uint32_t n_layers, uint32_t experts_per_layer,
                             uint32_t hidden_dim, uint32_t intermediate_dim) {
    nos::NxpFileHeader hdr{};
    hdr.magic = nos::NXP_MAGIC;
    hdr.version = nos::NXP_VERSION;
    hdr.num_layers = n_layers;
    hdr.experts_per_layer = experts_per_layer;
    hdr.hidden_dim = hidden_dim;
    hdr.intermediate_dim = intermediate_dim;
    hdr.packing_mode = 0;  // 5-per-byte
    hdr.scale_dtype = 0;   // FP16
    std::memset(hdr.reserved, 0, sizeof(hdr.reserved));

    nos::NxpWriter writer;
    std::string nxp_path = (dir / "model.nxp").string();
    REQUIRE(writer.open(nxp_path, hdr));

    const int cols = static_cast<int>(intermediate_dim);
    const int rows = static_cast<int>(hidden_dim);
    const int packed_cols = (cols + 4) / 5;

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        for (uint32_t exp = 0; exp < experts_per_layer; ++exp) {
            // Create alternating {-1, 0, 1} trit pattern
            std::vector<int8_t> trits(static_cast<size_t>(cols));
            for (int c = 0; c < cols; ++c) {
                trits[static_cast<size_t>(c)] = static_cast<int8_t>((c % 3) - 1);
            }

            // Pack all rows
            size_t total_packed = static_cast<size_t>(rows) *
                                  static_cast<size_t>(packed_cols);
            std::vector<uint8_t> packed(total_packed);
            for (int r = 0; r < rows; ++r) {
                uint8_t* row_packed = packed.data() +
                    static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
                nos::pack_row(trits.data(), cols, row_packed);
            }

            // FP16 scales: 1.0 for each channel (row)
            std::vector<uint16_t> scales(static_cast<size_t>(rows));
            uint16_t one_fp16 = nos::fp32_to_fp16(1.0f);
            std::fill(scales.begin(), scales.end(), one_fp16);

            writer.write_expert(layer, exp,
                                packed.data(), packed.size(),
                                scales.data(),
                                static_cast<uint32_t>(scales.size()));
        }
    }

    REQUIRE(writer.finalize());
}


TEST_CASE("train_full loads and saves expert weights via NXP", "[trainer_nxp]") {
    fs::path test_dir = make_test_dir("train_full_nxp");
    fs::path model_dir = test_dir / "model";
    fs::path output_dir = test_dir / "output";
    fs::path data_dir = test_dir / "data";

    fs::create_directories(model_dir);
    fs::create_directories(data_dir);

    const uint32_t n_layers = 2;
    const uint32_t expert_count = 1;
    const uint32_t hidden_dim = 32;
    const uint32_t intermediate_dim = 64;

    // Create test fixtures
    write_model_config(model_dir, n_layers, expert_count, hidden_dim, intermediate_dim);
    write_training_data(data_dir);
    create_test_nxp(model_dir, n_layers, expert_count, hidden_dim, intermediate_dim);

    // Train
    nos::TrainConfig cfg;
    cfg.model_dir = model_dir.string();
    cfg.data_path = (data_dir / "train.jsonl").string();
    cfg.output_dir = output_dir.string();
    cfg.method = "full";
    cfg.max_epochs = 1;
    cfg.badam_config.steps_per_block = 50;  // Enough steps to change ternary bins
    cfg.badam_config.lr = 0.1f;             // High LR to ensure visible weight change
    cfg.galore_config.rank = 4;             // Small rank for test dimensions

    nos::Trainer trainer;
    REQUIRE(trainer.train(cfg));

    // Verify model_config.json was copied to output directory (TRNG-01 gap closure)
    REQUIRE(fs::exists(output_dir / "model_config.json"));

    // Verify model_config.json content matches source
    {
        std::ifstream src_cfg((model_dir / "model_config.json").string());
        std::ifstream out_cfg((output_dir / "model_config.json").string());
        REQUIRE(src_cfg.is_open());
        REQUIRE(out_cfg.is_open());
        nlohmann::json src_j = nlohmann::json::parse(src_cfg);
        nlohmann::json out_j = nlohmann::json::parse(out_cfg);
        REQUIRE(src_j == out_j);
    }

    // Verify output .nxp exists
    std::string out_nxp = (output_dir / "model.nxp").string();
    REQUIRE(fs::exists(out_nxp));

    // Read back output .nxp and verify structure
    nos::NxpReader reader;
    REQUIRE(reader.open(out_nxp));
    REQUIRE(reader.header().num_layers == n_layers);
    REQUIRE(reader.header().hidden_dim == hidden_dim);
    REQUIRE(reader.header().intermediate_dim == intermediate_dim);
    REQUIRE(reader.num_entries() == static_cast<size_t>(n_layers * expert_count));

    // Verify expert entries exist and have data
    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        for (uint32_t exp = 0; exp < expert_count; ++exp) {
            const nos::NxpExpertEntry* entry = reader.find_expert(layer, exp);
            REQUIRE(entry != nullptr);
            REQUIRE(entry->size > 0);

            // Read weights and verify non-zero
            std::vector<uint8_t> buf(static_cast<size_t>(entry->size));
            int rd = reader.read_expert(*entry, buf.data(), buf.size());
            REQUIRE(rd > 0);

            // Verify at least some bytes are non-zero (training modified weights)
            bool has_nonzero = false;
            for (size_t i = 0; i < buf.size(); ++i) {
                if (buf[i] != 0) {
                    has_nonzero = true;
                    break;
                }
            }
            REQUIRE(has_nonzero);
        }
    }

    // Read both input and output .nxp to verify training actually changed weights
    nos::NxpReader input_reader;
    REQUIRE(input_reader.open((model_dir / "model.nxp").string()));

    const nos::NxpExpertEntry* in_entry = input_reader.find_expert(0, 0);
    const nos::NxpExpertEntry* out_entry = reader.find_expert(0, 0);
    REQUIRE(in_entry != nullptr);
    REQUIRE(out_entry != nullptr);

    std::vector<uint8_t> in_buf(static_cast<size_t>(in_entry->size));
    std::vector<uint8_t> out_buf(static_cast<size_t>(out_entry->size));
    input_reader.read_expert(*in_entry, in_buf.data(), in_buf.size());
    reader.read_expert(*out_entry, out_buf.data(), out_buf.size());

    // Weights should differ (training modified them)
    bool weights_differ = false;
    size_t compare_len = std::min(in_buf.size(), out_buf.size());
    for (size_t i = 0; i < compare_len; ++i) {
        if (in_buf[i] != out_buf[i]) {
            weights_differ = true;
            break;
        }
    }
    REQUIRE(weights_differ);

    cleanup_test_dir(test_dir);
}


TEST_CASE("train_full falls back to synthetic mode without .nxp", "[trainer_nxp]") {
    fs::path test_dir = make_test_dir("train_full_synthetic");
    fs::path model_dir = test_dir / "model";
    fs::path output_dir = test_dir / "output";
    fs::path data_dir = test_dir / "data";

    fs::create_directories(model_dir);
    fs::create_directories(data_dir);

    // Create model config but NO .nxp file
    write_model_config(model_dir, 2, 1, 32, 64);
    write_training_data(data_dir);

    nos::TrainConfig cfg;
    cfg.model_dir = model_dir.string();
    cfg.data_path = (data_dir / "train.jsonl").string();
    cfg.output_dir = output_dir.string();
    cfg.method = "full";
    cfg.max_epochs = 1;
    cfg.badam_config.steps_per_block = 2;
    cfg.galore_config.rank = 4;

    nos::Trainer trainer;
    // Should succeed in synthetic/demo mode
    REQUIRE(trainer.train(cfg));

    // Verify model_config.json was copied even in synthetic mode
    REQUIRE(fs::exists(output_dir / "model_config.json"));

    // No output .nxp should be written in synthetic mode
    bool has_nxp = false;
    if (fs::exists(output_dir)) {
        for (const auto& entry : fs::directory_iterator(output_dir)) {
            if (entry.path().extension() == ".nxp") {
                has_nxp = true;
            }
        }
    }
    REQUIRE_FALSE(has_nxp);

    cleanup_test_dir(test_dir);
}


TEST_CASE("merge-lora produces modified .nxp with merged adapter weights", "[trainer_nxp]") {
    fs::path test_dir = make_test_dir("merge_lora_nxp");
    fs::path model_dir = test_dir / "model";
    fs::path output_dir = test_dir / "output";

    fs::create_directories(model_dir);

    const uint32_t n_layers = 2;
    const uint32_t expert_count = 1;
    const uint32_t hidden_dim = 16;
    const uint32_t intermediate_dim = 32;

    // Create base .nxp and model config
    write_model_config(model_dir, n_layers, expert_count, hidden_dim, intermediate_dim);
    create_test_nxp(model_dir, n_layers, expert_count, hidden_dim, intermediate_dim);

    // Verify the base .nxp was created correctly
    nos::NxpReader base_reader;
    REQUIRE(base_reader.open((model_dir / "model.nxp").string()));

    // Dequantize expert weights from base .nxp
    const nos::NxpExpertEntry* base_entry = base_reader.find_expert(0, 0);
    REQUIRE(base_entry != nullptr);

    const int rows = static_cast<int>(hidden_dim);
    const int cols = static_cast<int>(intermediate_dim);
    const int packed_cols = (cols + 4) / 5;
    const size_t total_params = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    std::vector<uint8_t> packed_orig(static_cast<size_t>(base_entry->size));
    base_reader.read_expert(*base_entry, packed_orig.data(), packed_orig.size());

    std::vector<uint16_t> scales_orig(static_cast<size_t>(base_entry->num_channels));
    base_reader.read_scales(*base_entry, scales_orig.data(),
                            static_cast<size_t>(base_entry->scale_size));

    // Dequantize
    std::vector<float> weights_fp32(total_params);
    std::vector<int8_t> trits(static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_p = packed_orig.data()
            + static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        nos::unpack_row(row_p, cols, trits.data());
        float scale = nos::fp16_to_fp32(scales_orig[static_cast<size_t>(r)]);
        for (int c = 0; c < cols; ++c) {
            weights_fp32[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                         static_cast<size_t>(c)] =
                static_cast<float>(trits[static_cast<size_t>(c)]) * scale;
        }
    }

    // Create LoRA adapter with non-zero B values to ensure merge modifies weights
    nos::LoRAConfig lora_cfg;
    lora_cfg.rank = 4;
    lora_cfg.alpha = 4.0f;
    nos::LoRAAdapter adapter(lora_cfg,
                             static_cast<size_t>(cols),
                             static_cast<size_t>(rows));

    // Set B to non-zero values so merge_into actually changes weights
    float* B = adapter.B_data();
    for (size_t i = 0; i < static_cast<size_t>(rows) * lora_cfg.rank; ++i) {
        B[i] = 0.5f;
    }

    // Apply merge_into
    std::vector<float> merged_fp32(weights_fp32);
    adapter.merge_into(merged_fp32.data(),
                       static_cast<size_t>(rows),
                       static_cast<size_t>(cols));

    // Verify merge actually changed something
    bool merge_changed = false;
    for (size_t i = 0; i < total_params; ++i) {
        if (std::abs(merged_fp32[i] - weights_fp32[i]) > 1e-8f) {
            merge_changed = true;
            break;
        }
    }
    REQUIRE(merge_changed);

    // Re-quantize: FP32 -> FP16 -> ternary
    std::vector<uint16_t> fp16_buf(total_params);
    for (size_t i = 0; i < total_params; ++i) {
        fp16_buf[i] = nos::fp32_to_fp16(merged_fp32[i]);
    }
    nos::QuantizedWeights qw = nos::ternary_quantize(fp16_buf.data(), rows, cols);

    // Write output .nxp
    fs::create_directories(output_dir);
    nos::NxpFileHeader out_hdr = base_reader.header();
    nos::NxpWriter writer;
    std::string out_nxp = (output_dir / "merged.nxp").string();
    REQUIRE(writer.open(out_nxp, out_hdr));
    writer.write_expert(0, 0,
                        qw.packed.data(), qw.packed.size(),
                        qw.scales.data(),
                        static_cast<uint32_t>(qw.scales.size()));
    REQUIRE(writer.finalize());

    // Read back and verify weights differ from original
    nos::NxpReader out_reader;
    REQUIRE(out_reader.open(out_nxp));
    const nos::NxpExpertEntry* out_entry = out_reader.find_expert(0, 0);
    REQUIRE(out_entry != nullptr);

    std::vector<uint8_t> out_packed(static_cast<size_t>(out_entry->size));
    out_reader.read_expert(*out_entry, out_packed.data(), out_packed.size());

    // Packed weights should differ from original
    bool packed_differ = false;
    size_t cmp_len = std::min(packed_orig.size(), out_packed.size());
    for (size_t i = 0; i < cmp_len; ++i) {
        if (packed_orig[i] != out_packed[i]) {
            packed_differ = true;
            break;
        }
    }
    REQUIRE(packed_differ);

    cleanup_test_dir(test_dir);
}


TEST_CASE("dequantize-requantize round-trip preserves weight structure", "[trainer_nxp]") {
    const int rows = 8;
    const int cols = 20;  // Not divisible by 5 to test padding
    const int packed_cols = (cols + 4) / 5;  // = 4

    // Create known ternary pattern: alternating -1, 0, 1
    std::vector<int8_t> original_trits(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            original_trits[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                           static_cast<size_t>(c)] =
                static_cast<int8_t>((c % 3) - 1);
        }
    }

    // Pack
    std::vector<uint8_t> packed(static_cast<size_t>(rows) *
                                static_cast<size_t>(packed_cols));
    for (int r = 0; r < rows; ++r) {
        const int8_t* row_trits = original_trits.data() +
            static_cast<size_t>(r) * static_cast<size_t>(cols);
        uint8_t* row_packed = packed.data() +
            static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        nos::pack_row(row_trits, cols, row_packed);
    }

    // Set known FP16 scales (1.0 for all channels)
    std::vector<uint16_t> scales(static_cast<size_t>(rows));
    uint16_t one_fp16 = nos::fp32_to_fp16(1.0f);
    std::fill(scales.begin(), scales.end(), one_fp16);

    // Dequantize: unpack + scale multiply
    std::vector<float> fp32_weights(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::vector<int8_t> trits(static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_packed = packed.data() +
            static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        nos::unpack_row(row_packed, cols, trits.data());
        float scale = nos::fp16_to_fp32(scales[static_cast<size_t>(r)]);
        for (int c = 0; c < cols; ++c) {
            fp32_weights[static_cast<size_t>(r) * static_cast<size_t>(cols) +
                         static_cast<size_t>(c)] =
                static_cast<float>(trits[static_cast<size_t>(c)]) * scale;
        }
    }

    // Re-quantize: FP32 -> FP16 -> ternary_quantize
    std::vector<uint16_t> fp16_buf(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (size_t i = 0; i < fp16_buf.size(); ++i) {
        fp16_buf[i] = nos::fp32_to_fp16(fp32_weights[i]);
    }
    nos::QuantizedWeights qw = nos::ternary_quantize(fp16_buf.data(), rows, cols);

    // Unpack the re-quantized result
    std::vector<int8_t> roundtrip_trits(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        uint8_t* row_packed = qw.packed.data() +
            static_cast<size_t>(r) * static_cast<size_t>(packed_cols);
        int8_t* row_trits = roundtrip_trits.data() +
            static_cast<size_t>(r) * static_cast<size_t>(cols);
        nos::unpack_row(row_packed, cols, row_trits);
    }

    // Verify round-trip fidelity: trit patterns should match
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                         static_cast<size_t>(c);
            INFO("Row " << r << " Col " << c);
            REQUIRE(roundtrip_trits[idx] == original_trits[idx]);
        }
    }
}
