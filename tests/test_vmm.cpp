/// @file test_vmm.cpp
/// @brief VMM integration tests with synthetic .nxp files.

#include <catch2/catch_test_macros.hpp>

#include "vmm/vmm.h"
#include "format/expert_format.h"
#include "format/crc32.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

/// Align value up to 64-byte boundary.
size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

/// Create a temporary .nxp file with known expert data.
/// Each expert has weight_size bytes of weight data and num_channels FP16 scales.
/// Weight data for expert (layer_id, expert_id) is filled with
/// (layer_id * experts_per_layer + expert_id + 1) & 0xFF for verification.
struct TestNxp {
    std::string path;
    uint32_t num_layers;
    uint32_t experts_per_layer;
    size_t weight_size;
    uint32_t num_channels;
    size_t max_expert_size;  // weight_size aligned + scale bytes aligned

    TestNxp(uint32_t layers, uint32_t experts, size_t wsize, uint32_t channels)
        : num_layers(layers)
        , experts_per_layer(experts)
        , weight_size(wsize)
        , num_channels(channels) {
        // Generate temp file path
        char tmpl[] = "/tmp/test_vmm_XXXXXX";
        int fd = ::mkstemp(tmpl);
        REQUIRE(fd >= 0);
        ::close(fd);
        path = std::string(tmpl) + ".nxp";
        std::rename(tmpl, path.c_str());

        // Compute max_expert_size: aligned weight + aligned scale
        size_t scale_bytes = static_cast<size_t>(num_channels) * sizeof(uint16_t);
        max_expert_size = align_up(weight_size, 64) + align_up(scale_bytes, 64);

        // Write .nxp file
        nos::NxpFileHeader hdr{};
        hdr.num_layers = num_layers;
        hdr.experts_per_layer = experts_per_layer;
        hdr.hidden_dim = num_channels;
        hdr.intermediate_dim = num_channels;
        hdr.packing_mode = 0;
        hdr.scale_dtype = 0;

        nos::NxpWriter writer;
        REQUIRE(writer.open(path, hdr));

        for (uint32_t l = 0; l < num_layers; ++l) {
            for (uint32_t e = 0; e < experts_per_layer; ++e) {
                // Fill weights with deterministic pattern
                uint32_t dense_id = l * experts_per_layer + e;
                uint8_t fill_byte = static_cast<uint8_t>((dense_id + 1) & 0xFF);
                std::vector<uint8_t> weights(weight_size, fill_byte);

                // Fill scale factors with deterministic pattern
                std::vector<uint16_t> scales(num_channels);
                for (uint32_t c = 0; c < num_channels; ++c) {
                    scales[c] = static_cast<uint16_t>(dense_id * 100 + c);
                }

                writer.write_expert(l, e, weights.data(), weight_size,
                                    scales.data(), num_channels);
            }
        }

        REQUIRE(writer.finalize());
    }

    ~TestNxp() {
        std::remove(path.c_str());
    }

    TestNxp(const TestNxp&) = delete;
    TestNxp& operator=(const TestNxp&) = delete;
};

}  // anonymous namespace

TEST_CASE("VMM: get_handle returns correct index and generation", "[vmm]") {
    TestNxp nxp(2, 4, 1024, 32);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 8;  // All experts fit
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    // Verify handle indices
    auto h00 = vmm.get_handle(0, 0);
    CHECK(h00.index == 0);
    CHECK(h00.generation == 0);

    auto h13 = vmm.get_handle(1, 3);
    CHECK(h13.index == 1 * 4 + 3);
    CHECK(h13.generation == 0);

    // Out of bounds
    auto hbad = vmm.get_handle(5, 0);
    CHECK(hbad == nos::INVALID_HANDLE);
}

TEST_CASE("VMM: pin/unpin round-trip with data verification", "[vmm]") {
    TestNxp nxp(2, 4, 1024, 32);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 8;
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    auto handle = vmm.get_handle(0, 0);
    CHECK(vmm.page_state(handle) == nos::PageState::EVICTED);

    // Pin: triggers load from NVMe
    const uint8_t* data = vmm.pin(handle);
    REQUIRE(data != nullptr);
    CHECK(vmm.page_state(handle) == nos::PageState::CACHED);

    // Verify data matches what was written
    // Expert (0,0) has dense_id=0, fill_byte=1
    for (size_t i = 0; i < nxp.weight_size; ++i) {
        if (data[i] != 1) {
            INFO("Mismatch at byte " << i << ": expected 1, got " << (int)data[i]);
            CHECK(data[i] == 1);
            break;
        }
    }

    // Unpin
    vmm.unpin(handle);
    CHECK(vmm.page_state(handle) == nos::PageState::RESIDENT);
}

TEST_CASE("VMM: stale handle detection after eviction", "[vmm]") {
    // Small cache: 4 slots, 8 experts -- forces eviction
    TestNxp nxp(2, 4, 512, 16);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 4;  // Only 4 slots
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    // Get handle to expert (0,0)
    auto old_handle = vmm.get_handle(0, 0);

    // Pin and unpin expert (0,0)
    const uint8_t* data = vmm.pin(old_handle);
    REQUIRE(data != nullptr);
    vmm.unpin(old_handle);

    // Load 4 more experts to fill cache and force eviction of (0,0)
    for (uint32_t e = 1; e < 5; ++e) {
        uint32_t layer = e / nxp.experts_per_layer;
        uint32_t expert = e % nxp.experts_per_layer;
        auto h = vmm.get_handle(layer, expert);
        const uint8_t* d = vmm.pin(h);
        REQUIRE(d != nullptr);
        vmm.unpin(h);
    }

    // Old handle should be stale (generation mismatch)
    const uint8_t* stale_data = vmm.pin(old_handle);
    CHECK(stale_data == nullptr);

    // Getting a fresh handle should work
    auto new_handle = vmm.get_handle(0, 0);
    CHECK(new_handle.generation > old_handle.generation);
    const uint8_t* fresh_data = vmm.pin(new_handle);
    REQUIRE(fresh_data != nullptr);

    // Verify data is still correct after reload
    for (size_t i = 0; i < nxp.weight_size; ++i) {
        if (fresh_data[i] != 1) {
            INFO("Mismatch at byte " << i);
            CHECK(fresh_data[i] == 1);
            break;
        }
    }

    vmm.unpin(new_handle);
}

TEST_CASE("VMM: multiple pin on same expert increments refcount", "[vmm]") {
    TestNxp nxp(1, 4, 512, 16);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 4;
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    auto handle = vmm.get_handle(0, 0);

    // Pin twice
    const uint8_t* data1 = vmm.pin(handle);
    REQUIRE(data1 != nullptr);
    const uint8_t* data2 = vmm.pin(handle);
    REQUIRE(data2 != nullptr);
    CHECK(data1 == data2);  // Same pointer
    CHECK(vmm.page_state(handle) == nos::PageState::CACHED);

    // Unpin once -- still CACHED (refcount > 0)
    vmm.unpin(handle);
    CHECK(vmm.page_state(handle) == nos::PageState::CACHED);

    // Unpin again -- now RESIDENT (refcount == 0)
    vmm.unpin(handle);
    CHECK(vmm.page_state(handle) == nos::PageState::RESIDENT);
}

TEST_CASE("VMM: eviction under pressure with small cache", "[vmm]") {
    // 4 cache slots, 8 experts
    TestNxp nxp(2, 4, 512, 16);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 4;
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    // Load all 8 experts sequentially (forces eviction)
    for (uint32_t l = 0; l < nxp.num_layers; ++l) {
        for (uint32_t e = 0; e < nxp.experts_per_layer; ++e) {
            auto h = vmm.get_handle(l, e);
            const uint8_t* data = vmm.pin(h);
            REQUIRE(data != nullptr);

            // Verify data
            uint32_t dense_id = l * nxp.experts_per_layer + e;
            uint8_t expected = static_cast<uint8_t>((dense_id + 1) & 0xFF);
            CHECK(data[0] == expected);

            vmm.unpin(h);
        }
    }

    // Should have at most 4 resident + some evicted
    CHECK(vmm.resident_count() <= 4);
}

TEST_CASE("VMM: CRC32 verification detects corruption", "[vmm]") {
    TestNxp nxp(1, 2, 512, 16);

    // Corrupt the .nxp file: flip a byte in the first expert's weight data
    {
        int fd = ::open(nxp.path.c_str(), O_RDWR);
        REQUIRE(fd >= 0);

        // Read header to find first expert's data offset
        nos::NxpReader reader;
        REQUIRE(reader.open(nxp.path));
        const nos::NxpExpertEntry* entry = reader.find_expert(0, 0);
        REQUIRE(entry != nullptr);
        uint64_t offset = entry->offset;
        reader.close();

        // Read one byte from the weight data region and flip it
        uint8_t byte = 0;
        ::pread(fd, &byte, 1, static_cast<off_t>(offset + 10));
        byte = static_cast<uint8_t>(~byte);
        ::pwrite(fd, &byte, 1, static_cast<off_t>(offset + 10));
        ::close(fd);
    }

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 4;
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    // Pin the corrupted expert -- should fail CRC check
    auto handle = vmm.get_handle(0, 0);
    const uint8_t* data = vmm.pin(handle);
    CHECK(data == nullptr);  // CRC mismatch

    // Expert 1 should still work (not corrupted)
    auto handle1 = vmm.get_handle(0, 1);
    const uint8_t* data1 = vmm.pin(handle1);
    CHECK(data1 != nullptr);
    vmm.unpin(handle1);
}

TEST_CASE("VMM: cache hit rate > 85% with temporal locality", "[vmm]") {
    // 4 layers x 8 experts = 32 experts, cache fits 24 (75%).
    // Under Zipf(1.0), top 24 of 32 experts account for ~93% of accesses,
    // giving CLOCK-Pro enough headroom to exceed 85% after its warmup phase.
    TestNxp nxp(4, 8, 256, 8);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 24;
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    // Generate Zipf-like access pattern with temporal locality
    std::mt19937 rng(42);
    constexpr size_t NUM_ACCESSES = 2000;
    constexpr size_t TOTAL_EXPERTS = 32;

    // Zipf CDF
    std::vector<double> cdf(TOTAL_EXPERTS);
    double sum = 0.0;
    for (size_t k = 0; k < TOTAL_EXPERTS; ++k) {
        sum += 1.0 / static_cast<double>(k + 1);
        cdf[k] = sum;
    }
    for (auto& v : cdf) v /= sum;

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Measure cache performance using load_count (misses = loads from disk)
    size_t loads_before = vmm.load_count();

    for (size_t i = 0; i < NUM_ACCESSES; ++i) {
        double r = dist(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        uint32_t dense_id = static_cast<uint32_t>(
            std::distance(cdf.begin(), it));
        if (dense_id >= TOTAL_EXPERTS) dense_id = TOTAL_EXPERTS - 1;

        uint32_t layer = dense_id / nxp.experts_per_layer;
        uint32_t expert = dense_id % nxp.experts_per_layer;

        auto handle = vmm.get_handle(layer, expert);
        REQUIRE(handle != nos::INVALID_HANDLE);

        const uint8_t* data = vmm.pin(handle);
        REQUIRE(data != nullptr);
        vmm.unpin(handle);
    }

    size_t loads_after = vmm.load_count();
    size_t misses = loads_after - loads_before;
    double hit_rate = 1.0 - static_cast<double>(misses) / static_cast<double>(NUM_ACCESSES);
    INFO("VMM cache hit rate: " << (hit_rate * 100.0) << "%"
         << " (" << misses << " misses / " << NUM_ACCESSES << " accesses)");
    CHECK(hit_rate > 0.85);
}

TEST_CASE("VMM: no duplicate loads within 100-access window", "[vmm]") {
    // 2 layers x 4 experts = 8 experts, cache fits all 8
    TestNxp nxp(2, 4, 256, 8);

    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * 8;  // All fit
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;

    nos::Vmm vmm(config);

    // First: load all 8 experts
    for (uint32_t l = 0; l < nxp.num_layers; ++l) {
        for (uint32_t e = 0; e < nxp.experts_per_layer; ++e) {
            auto h = vmm.get_handle(l, e);
            const uint8_t* d = vmm.pin(h);
            REQUIRE(d != nullptr);
            vmm.unpin(h);
        }
    }

    size_t loads_before = vmm.load_count();

    // Access all 8 experts 100 times -- no new loads should occur
    for (int round = 0; round < 100; ++round) {
        for (uint32_t l = 0; l < nxp.num_layers; ++l) {
            for (uint32_t e = 0; e < nxp.experts_per_layer; ++e) {
                auto h = vmm.get_handle(l, e);
                const uint8_t* d = vmm.pin(h);
                REQUIRE(d != nullptr);
                vmm.unpin(h);
            }
        }
    }

    size_t loads_after = vmm.load_count();
    CHECK(loads_after == loads_before);  // No new loads
}
