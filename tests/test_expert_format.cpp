/// @file test_expert_format.cpp
/// @brief NXP expert format write/read round-trip, corruption, alignment tests.
///
/// Tests: write-read round-trip, header validation, 64-byte alignment,
/// CRC integrity, retry logic, empty experts, large experts, find_expert,
/// and invalid file handling.

#include <catch2/catch_test_macros.hpp>

#include "expert_format.h"
#include "crc32.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Counter for unique temp file names
static int g_temp_counter = 0;

// Helper: create a temp file path that cleans up on scope exit
struct TempFile {
    std::string path;

    TempFile() {
        auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        auto tmp = fs::temp_directory_path() / ("nxp_test_" +
            std::to_string(ts) + "_" + std::to_string(g_temp_counter++) + ".nxp");
        path = tmp.string();
    }

    ~TempFile() {
        std::remove(path.c_str());
    }
};

// Helper: create a default NxpFileHeader for testing
static nos::NxpFileHeader make_test_header(uint32_t num_layers = 2,
                                            uint32_t experts_per_layer = 4,
                                            uint32_t hidden_dim = 4096,
                                            uint32_t intermediate_dim = 11008) {
    nos::NxpFileHeader hdr{};
    hdr.magic = nos::NXP_MAGIC;
    hdr.version = nos::NXP_VERSION;
    hdr.num_layers = num_layers;
    hdr.experts_per_layer = experts_per_layer;
    hdr.hidden_dim = hidden_dim;
    hdr.intermediate_dim = intermediate_dim;
    hdr.packing_mode = 0;  // 5-per-byte
    hdr.scale_dtype = 0;   // FP16
    std::memset(hdr.reserved, 0, sizeof(hdr.reserved));
    return hdr;
}

TEST_CASE("NXP write-read round-trip", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(1, 3);

    // Generate test data for 3 experts
    struct ExpertData {
        uint32_t layer_id;
        uint32_t expert_id;
        std::vector<uint8_t> weights;
        std::vector<uint16_t> scales;
    };

    std::mt19937 rng(42);
    std::vector<ExpertData> experts;
    for (uint32_t i = 0; i < 3; i++) {
        ExpertData e;
        e.layer_id = 0;
        e.expert_id = i;

        // Varying sizes: 100, 500, 1000 bytes
        size_t weight_sz = 100 + static_cast<size_t>(i) * 400;
        e.weights.resize(weight_sz);
        std::generate(e.weights.begin(), e.weights.end(), [&]() {
            return static_cast<uint8_t>(rng());
        });

        // 8 channels of FP16 scale factors
        e.scales.resize(8);
        std::generate(e.scales.begin(), e.scales.end(), [&]() {
            return static_cast<uint16_t>(rng() & 0xFFFF);
        });

        experts.push_back(std::move(e));
    }

    // Write
    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));

        for (const auto& e : experts) {
            writer.write_expert(e.layer_id, e.expert_id,
                                e.weights.data(), e.weights.size(),
                                e.scales.data(), static_cast<uint32_t>(e.scales.size()));
        }

        REQUIRE(writer.finalize());
    }

    // Read and verify
    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));
        REQUIRE(reader.num_entries() == 3);

        for (const auto& e : experts) {
            const auto* entry = reader.find_expert(e.layer_id, e.expert_id);
            REQUIRE(entry != nullptr);

            // Read weights
            std::vector<uint8_t> buf(static_cast<size_t>(entry->size));
            int n = reader.read_expert(*entry, buf.data(), buf.size());
            REQUIRE(n == static_cast<int>(e.weights.size()));
            REQUIRE(buf == e.weights);

            // Read scale factors
            std::vector<uint16_t> scale_buf(entry->num_channels);
            int sn = reader.read_scales(*entry, scale_buf.data(),
                                         static_cast<size_t>(entry->scale_size));
            REQUIRE(sn == static_cast<int>(entry->scale_size));
            REQUIRE(scale_buf == e.scales);
        }

        reader.close();
    }
}

TEST_CASE("NXP header validation", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(4, 8, 4096, 11008);

    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));
        // Write one expert so finalize succeeds with index
        uint8_t data[64]{};
        uint16_t scales[4] = {0x3C00, 0x3C00, 0x3C00, 0x3C00};
        writer.write_expert(0, 0, data, sizeof(data), scales, 4);
        REQUIRE(writer.finalize());
    }

    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));

        const auto& h = reader.header();
        REQUIRE(h.magic == nos::NXP_MAGIC);
        REQUIRE(h.version == nos::NXP_VERSION);
        REQUIRE(h.num_layers == 4);
        REQUIRE(h.experts_per_layer == 8);
        REQUIRE(h.hidden_dim == 4096);
        REQUIRE(h.intermediate_dim == 11008);
        REQUIRE(h.total_experts == 1);

        reader.close();
    }
}

TEST_CASE("NXP 64-byte alignment", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(1, 3);

    std::vector<nos::NxpExpertEntry> written_entries;
    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));

        // Write experts with non-aligned sizes to test padding
        uint8_t data1[37]{1};   // Not a multiple of 64
        uint16_t s1[2] = {100, 200};
        written_entries.push_back(
            writer.write_expert(0, 0, data1, sizeof(data1), s1, 2));

        uint8_t data2[100]{2};
        uint16_t s2[4] = {300, 400, 500, 600};
        written_entries.push_back(
            writer.write_expert(0, 1, data2, sizeof(data2), s2, 4));

        uint8_t data3[1]{3};
        uint16_t s3[1] = {700};
        written_entries.push_back(
            writer.write_expert(0, 2, data3, sizeof(data3), s3, 1));

        REQUIRE(writer.finalize());
    }

    // Verify alignment of all expert data offsets
    for (const auto& entry : written_entries) {
        REQUIRE(entry.offset % nos::NXP_ALIGNMENT == 0);
        REQUIRE(entry.scale_offset % nos::NXP_ALIGNMENT == 0);
    }

    // Also verify via reader
    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));

        for (uint32_t i = 0; i < 3; i++) {
            const auto* entry = reader.find_expert(0, i);
            REQUIRE(entry != nullptr);
            REQUIRE(entry->offset % nos::NXP_ALIGNMENT == 0);
            REQUIRE(entry->scale_offset % nos::NXP_ALIGNMENT == 0);
        }

        reader.close();
    }
}

TEST_CASE("NXP CRC integrity -- corruption detected", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(1, 1);

    // Write one expert with known data
    std::vector<uint8_t> weights(256, 0xAA);
    uint16_t scales[4] = {0x3C00, 0x3C00, 0x3C00, 0x3C00};

    nos::NxpExpertEntry written_entry{};
    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));
        written_entry = writer.write_expert(0, 0, weights.data(), weights.size(), scales, 4);
        REQUIRE(writer.finalize());
    }

    // First, verify it reads correctly (CRC passes)
    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));
        const auto* entry = reader.find_expert(0, 0);
        REQUIRE(entry != nullptr);

        std::vector<uint8_t> buf(static_cast<size_t>(entry->size));
        int n = reader.read_expert(*entry, buf.data(), buf.size());
        REQUIRE(n > 0);  // CRC passes
        reader.close();
    }

    // Corrupt one byte in the weight data
    {
        std::fstream file(tmp.path, std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(file.is_open());
        file.seekp(static_cast<std::streamoff>(written_entry.offset));
        char corrupted_byte = 0x55;  // Different from 0xAA
        file.write(&corrupted_byte, 1);
        file.close();
    }

    // Now read should fail (CRC mismatch after retries)
    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));
        const auto* entry = reader.find_expert(0, 0);
        REQUIRE(entry != nullptr);

        std::vector<uint8_t> buf(static_cast<size_t>(entry->size));
        int n = reader.read_expert(*entry, buf.data(), buf.size(), 3);
        REQUIRE(n == -1);  // All retries exhausted
        reader.close();
    }
}

TEST_CASE("NXP CRC retry -- permanent corruption exhausts retries", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(1, 1);

    std::vector<uint8_t> weights(128, 0xBB);
    uint16_t scales[2] = {0x3C00, 0x3C00};

    nos::NxpExpertEntry written_entry{};
    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));
        written_entry = writer.write_expert(0, 0, weights.data(), weights.size(), scales, 2);
        REQUIRE(writer.finalize());
    }

    // Corrupt the data
    {
        std::fstream file(tmp.path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(written_entry.offset + 10));
        char bad = 0x00;
        file.write(&bad, 1);
        file.close();
    }

    // Verify retry_count=0 still attempts once (0 retries = 1 attempt total)
    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));
        const auto* entry = reader.find_expert(0, 0);
        REQUIRE(entry != nullptr);

        std::vector<uint8_t> buf(static_cast<size_t>(entry->size));
        int n = reader.read_expert(*entry, buf.data(), buf.size(), 0);
        REQUIRE(n == -1);
        reader.close();
    }
}

TEST_CASE("NXP empty expert (0-byte weight data)", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(1, 1);

    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));
        // Write expert with 0-byte weight data and 0 channels
        writer.write_expert(0, 0, nullptr, 0, nullptr, 0);
        REQUIRE(writer.finalize());
    }

    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));
        REQUIRE(reader.num_entries() == 1);

        const auto* entry = reader.find_expert(0, 0);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->size == 0);
        REQUIRE(entry->scale_size == 0);

        // Reading 0 bytes should succeed (CRC of empty data matches)
        std::vector<uint8_t> buf(1);  // Minimal buffer
        int n = reader.read_expert(*entry, buf.data(), buf.size());
        REQUIRE(n == 0);

        reader.close();
    }
}

TEST_CASE("NXP large expert (1MB random data)", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(1, 1);

    constexpr size_t kWeightSize = 1024 * 1024;  // 1MB
    constexpr uint32_t kNumChannels = 4096;

    std::vector<uint8_t> weights(kWeightSize);
    std::mt19937 rng(777);
    std::generate(weights.begin(), weights.end(), [&]() {
        return static_cast<uint8_t>(rng());
    });

    std::vector<uint16_t> scales(kNumChannels);
    std::generate(scales.begin(), scales.end(), [&]() {
        return static_cast<uint16_t>(rng() & 0xFFFF);
    });

    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));
        writer.write_expert(0, 0, weights.data(), weights.size(),
                            scales.data(), kNumChannels);
        REQUIRE(writer.finalize());
    }

    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));

        const auto* entry = reader.find_expert(0, 0);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->size == kWeightSize);
        REQUIRE(entry->num_channels == kNumChannels);

        std::vector<uint8_t> buf(kWeightSize);
        int n = reader.read_expert(*entry, buf.data(), buf.size());
        REQUIRE(n == static_cast<int>(kWeightSize));
        REQUIRE(buf == weights);

        std::vector<uint16_t> scale_buf(kNumChannels);
        int sn = reader.read_scales(*entry, scale_buf.data(),
                                     static_cast<size_t>(entry->scale_size));
        REQUIRE(sn > 0);
        REQUIRE(scale_buf == scales);

        reader.close();
    }
}

TEST_CASE("NXP find_expert lookup", "[expert_format]") {
    TempFile tmp;
    auto hdr = make_test_header(2, 2);

    {
        nos::NxpWriter writer;
        REQUIRE(writer.open(tmp.path, hdr));

        uint8_t data[16]{};
        uint16_t scales[1] = {0};
        // Write experts at specific (layer, expert) positions
        writer.write_expert(0, 0, data, sizeof(data), scales, 1);
        writer.write_expert(0, 1, data, sizeof(data), scales, 1);
        writer.write_expert(1, 0, data, sizeof(data), scales, 1);
        writer.write_expert(1, 1, data, sizeof(data), scales, 1);

        REQUIRE(writer.finalize());
    }

    {
        nos::NxpReader reader;
        REQUIRE(reader.open(tmp.path));
        REQUIRE(reader.num_entries() == 4);

        // Found entries
        REQUIRE(reader.find_expert(0, 0) != nullptr);
        REQUIRE(reader.find_expert(0, 1) != nullptr);
        REQUIRE(reader.find_expert(1, 0) != nullptr);
        REQUIRE(reader.find_expert(1, 1) != nullptr);

        // Verify correct layer/expert IDs
        REQUIRE(reader.find_expert(0, 0)->layer_id == 0);
        REQUIRE(reader.find_expert(0, 0)->expert_id == 0);
        REQUIRE(reader.find_expert(1, 1)->layer_id == 1);
        REQUIRE(reader.find_expert(1, 1)->expert_id == 1);

        // Non-existent experts
        REQUIRE(reader.find_expert(2, 0) == nullptr);
        REQUIRE(reader.find_expert(0, 2) == nullptr);
        REQUIRE(reader.find_expert(99, 99) == nullptr);

        reader.close();
    }
}

TEST_CASE("NXP invalid file -- wrong magic", "[expert_format]") {
    TempFile tmp;

    // Write a file with wrong magic
    {
        std::ofstream f(tmp.path, std::ios::binary);
        nos::NxpFileHeader bad_hdr{};
        bad_hdr.magic = 0xDEADBEEF;  // Wrong magic
        bad_hdr.version = nos::NXP_VERSION;
        f.write(reinterpret_cast<const char*>(&bad_hdr), sizeof(bad_hdr));
        f.close();
    }

    nos::NxpReader reader;
    REQUIRE_FALSE(reader.open(tmp.path));
}

TEST_CASE("NXP invalid file -- non-existent file", "[expert_format]") {
    nos::NxpReader reader;
    REQUIRE_FALSE(reader.open("/tmp/this_file_does_not_exist_at_all.nxp"));
}
