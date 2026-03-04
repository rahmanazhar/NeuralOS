/// @file test_async_vmm.cpp
/// @brief Async VMM tests: pin_async, await_pin, prefetch_expert.

#include <catch2/catch_test_macros.hpp>

#include "vmm/vmm.h"
#include "format/expert_format.h"
#include "format/crc32.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

/// Align value up to 64-byte boundary.
size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

/// Create a temporary .nxp file with known expert data.
struct TestNxp {
    std::string path;
    uint32_t num_layers;
    uint32_t experts_per_layer;
    size_t weight_size;
    uint32_t num_channels;
    size_t max_expert_size;

    TestNxp(uint32_t layers, uint32_t experts, size_t wsize, uint32_t channels)
        : num_layers(layers)
        , experts_per_layer(experts)
        , weight_size(wsize)
        , num_channels(channels) {
        char tmpl[] = "/tmp/test_async_vmm_XXXXXX";
        int fd = ::mkstemp(tmpl);
        REQUIRE(fd >= 0);
        ::close(fd);
        path = std::string(tmpl) + ".nxp";
        std::rename(tmpl, path.c_str());

        size_t scale_bytes = static_cast<size_t>(num_channels) * sizeof(uint16_t);
        max_expert_size = align_up(weight_size, 64) + align_up(scale_bytes, 64);

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
                uint32_t dense_id = l * experts_per_layer + e;
                uint8_t fill_byte = static_cast<uint8_t>((dense_id + 1) & 0xFF);
                std::vector<uint8_t> weights(weight_size, fill_byte);

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

nos::Vmm make_vmm(const TestNxp& nxp) {
    nos::VmmConfig config{};
    config.expert_cache_bytes = nxp.max_expert_size * nxp.num_layers * nxp.experts_per_layer;
    config.max_expert_size = nxp.max_expert_size;
    config.num_layers = nxp.num_layers;
    config.experts_per_layer = nxp.experts_per_layer;
    config.nxp_path = nxp.path;
    return nos::Vmm(config);
}

}  // anonymous namespace

TEST_CASE("Async VMM: pin_async then await_pin loads expert", "[async_vmm]") {
    TestNxp nxp(2, 4, 1024, 32);
    auto vmm = make_vmm(nxp);

    auto handle = vmm.get_handle(0, 0);
    REQUIRE(handle != nos::INVALID_HANDLE);

    CHECK(vmm.page_state(handle) == nos::PageState::EVICTED);

    // Non-blocking pin_async
    bool ok = vmm.pin_async(handle);
    CHECK(ok);

    // Blocking await_pin
    const uint8_t* data = vmm.await_pin(handle);
    REQUIRE(data != nullptr);
    CHECK(vmm.page_state(handle) == nos::PageState::CACHED);

    // Verify data: expert (0,0) dense_id=0, fill_byte=1
    CHECK(data[0] == 1);
    CHECK(data[1023] == 1);

    vmm.unpin(handle);
}

TEST_CASE("Async VMM: prefetch_expert for non-existent expert does not crash", "[async_vmm]") {
    TestNxp nxp(2, 4, 1024, 32);
    auto vmm = make_vmm(nxp);

    // Out-of-bounds layer: should not crash
    vmm.prefetch_expert(100, 0);

    // Out-of-bounds expert: should not crash
    vmm.prefetch_expert(0, 100);

    // Both invalid: should not crash
    vmm.prefetch_expert(UINT32_MAX, UINT32_MAX);
}

TEST_CASE("Async VMM: concurrent pin_async for same expert loads only once", "[async_vmm]") {
    TestNxp nxp(2, 4, 1024, 32);
    auto vmm = make_vmm(nxp);

    auto handle = vmm.get_handle(1, 2);
    REQUIRE(handle != nos::INVALID_HANDLE);

    size_t load_before = vmm.load_count();

    // Multiple pin_async calls for the same expert
    vmm.pin_async(handle);
    vmm.pin_async(handle);
    vmm.pin_async(handle);

    // Await to ensure load completes
    const uint8_t* data = vmm.await_pin(handle);
    REQUIRE(data != nullptr);

    // Only one load should have occurred (CAS ensures only one thread wins)
    size_t load_after = vmm.load_count();
    CHECK(load_after - load_before == 1);

    vmm.unpin(handle);
}

TEST_CASE("Async VMM: pin_async + await_pin returns same data as synchronous pin", "[async_vmm]") {
    TestNxp nxp(2, 4, 1024, 32);

    // Load via synchronous pin
    {
        auto vmm = make_vmm(nxp);
        auto handle = vmm.get_handle(0, 1);
        const uint8_t* sync_data = vmm.pin(handle);
        REQUIRE(sync_data != nullptr);

        // Expert (0,1) dense_id=1, fill_byte=2
        CHECK(sync_data[0] == 2);
        CHECK(sync_data[511] == 2);
        vmm.unpin(handle);
    }

    // Load via async pin
    {
        auto vmm = make_vmm(nxp);
        auto handle = vmm.get_handle(0, 1);
        vmm.pin_async(handle);
        const uint8_t* async_data = vmm.await_pin(handle);
        REQUIRE(async_data != nullptr);

        // Same data
        CHECK(async_data[0] == 2);
        CHECK(async_data[511] == 2);
        vmm.unpin(handle);
    }
}

TEST_CASE("Async VMM: prefetch_expert followed by pin returns immediately (cache hit)", "[async_vmm]") {
    TestNxp nxp(2, 4, 1024, 32);
    auto vmm = make_vmm(nxp);

    // Prefetch expert (1, 3)
    vmm.prefetch_expert(1, 3);

    // Wait a bit for async load to complete
    // (On macOS with pread backend, this completes very fast)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto handle = vmm.get_handle(1, 3);
    REQUIRE(handle != nos::INVALID_HANDLE);

    // Pin should be a cache hit (already loaded by prefetch)
    auto state_before = vmm.page_state(handle);
    CHECK(state_before == nos::PageState::RESIDENT);

    const uint8_t* data = vmm.pin(handle);
    REQUIRE(data != nullptr);

    // Expert (1,3) dense_id=7, fill_byte=8
    CHECK(data[0] == 8);

    // Should have been a cache hit
    auto stats = vmm.stats();
    CHECK(stats.cache_hits >= 1);

    vmm.unpin(handle);
}

TEST_CASE("Async VMM: pin_async is idempotent on already-resident expert", "[async_vmm]") {
    TestNxp nxp(2, 4, 1024, 32);
    auto vmm = make_vmm(nxp);

    auto handle = vmm.get_handle(0, 0);

    // First: synchronous pin to make it RESIDENT/CACHED
    const uint8_t* data = vmm.pin(handle);
    REQUIRE(data != nullptr);
    vmm.unpin(handle);  // Now RESIDENT

    size_t load_before = vmm.load_count();

    // pin_async on already-RESIDENT: should succeed without re-loading
    bool ok = vmm.pin_async(handle);
    CHECK(ok);

    size_t load_after = vmm.load_count();
    CHECK(load_after == load_before);  // No additional load
}
