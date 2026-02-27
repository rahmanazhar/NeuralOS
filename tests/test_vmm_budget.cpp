/// @file test_vmm_budget.cpp
/// @brief Integration tests for budget-aware VMM: Vmm::create() factory,
///        KV cache allocation, VmmStats tracking.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "vmm/vmm.h"
#include "vmm/memory_budget.h"
#include "format/expert_format.h"
#include "format/crc32.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using Catch::Matchers::ContainsSubstring;

namespace {

/// Align value up to 64-byte boundary.
size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

constexpr size_t GB = size_t(1) << 30;
constexpr size_t MB = size_t(1) << 20;

/// Create a temporary .nxp file with known dimensions.
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
        char tmpl[] = "/tmp/test_vmm_budget_XXXXXX";
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

    ~TestNxp() { std::remove(path.c_str()); }
    TestNxp(const TestNxp&) = delete;
    TestNxp& operator=(const TestNxp&) = delete;
};

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Sufficient budget -- Vmm::create() returns non-null
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: sufficient budget creates VMM", "[vmm_budget]") {
    // 4 layers * 8 experts, expert_size ~1KB each. Budget = 512MB.
    TestNxp nxp(4, 8, 1024, 32);

    nos::VmmFullConfig cfg{};
    cfg.nxp_path = nxp.path;
    cfg.user_budget_bytes = 512 * MB;
    cfg.desired_context_length = 512;

    auto vmm = nos::Vmm::create(cfg);
    REQUIRE(vmm != nullptr);

    CHECK(vmm->budget().sufficient == true);
    CHECK(vmm->budget().expert_slots > 0);
    CHECK(vmm->budget().total == 512 * MB);
}

// ═══════════════════════════════════════════════════════════════════════════
// Insufficient budget -- Vmm::create() returns nullptr
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: insufficient budget returns nullptr", "[vmm_budget]") {
    TestNxp nxp(4, 8, 1024, 32);

    nos::VmmFullConfig cfg{};
    cfg.nxp_path = nxp.path;
    cfg.user_budget_bytes = 1 * MB;  // Way too small (< os_overhead alone)
    cfg.desired_context_length = 512;

    auto vmm = nos::Vmm::create(cfg);
    CHECK(vmm == nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════
// Budget partitioning matches formula
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: partition sums match total", "[vmm_budget]") {
    TestNxp nxp(4, 8, 1024, 32);

    nos::VmmFullConfig cfg{};
    cfg.nxp_path = nxp.path;
    cfg.user_budget_bytes = 512 * MB;
    cfg.desired_context_length = 512;

    auto vmm = nos::Vmm::create(cfg);
    REQUIRE(vmm != nullptr);

    const auto& bp = vmm->budget();
    size_t sum = bp.expert_cache + bp.kv_cache + bp.working_buffers + bp.os_overhead;
    // Sum should be <= total; difference is the fractional slot remainder
    CHECK(sum <= bp.total);
    CHECK(bp.total - sum < nxp.max_expert_size);
}

// ═══════════════════════════════════════════════════════════════════════════
// KV cache allocated
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: KV cache allocated", "[vmm_budget]") {
    TestNxp nxp(4, 8, 1024, 32);

    nos::VmmFullConfig cfg{};
    cfg.nxp_path = nxp.path;
    cfg.user_budget_bytes = 512 * MB;
    cfg.desired_context_length = 512;

    auto vmm = nos::Vmm::create(cfg);
    REQUIRE(vmm != nullptr);

    CHECK(vmm->kv_cache_base() != nullptr);
    CHECK(vmm->kv_cache_size() == vmm->budget().kv_cache);
    CHECK(vmm->kv_cache_size() > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Statistics tracking -- cache misses and evictions
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: statistics track pins, misses, evictions", "[vmm_budget]") {
    // 2 layers * 4 experts = 8 experts, cache fits only 4 slots
    TestNxp nxp(2, 4, 512, 16);

    nos::VmmFullConfig cfg{};
    cfg.nxp_path = nxp.path;
    // Budget: 512MB overhead + working + kv + 4 expert slots
    // The budget formula will compute expert_cache from the remainder
    // We want ~4 slots: expert_cache ~ 4 * max_expert_size
    size_t os = 512 * MB;
    size_t working = (256 * sizeof(float) * 3 * 2 + 63) & ~size_t(63);
    // kv_per_token for these defaults (n_kv_heads=8, head_dim=128): small
    // But we're using VmmFullConfig defaults, so we set budget conservatively
    cfg.nxp_path = nxp.path;
    cfg.user_budget_bytes = os + working + 4 * nxp.max_expert_size + 1 * MB;
    cfg.desired_context_length = 32;

    auto vmm = nos::Vmm::create(cfg);
    REQUIRE(vmm != nullptr);

    // Check initial stats
    auto s0 = vmm->stats();
    CHECK(s0.total_pins == 0);
    CHECK(s0.cache_hits == 0);
    CHECK(s0.cache_misses == 0);
    CHECK(s0.evictions == 0);

    // Access all 8 experts -- first 4 are cold misses, rest force evictions
    for (uint32_t l = 0; l < nxp.num_layers; ++l) {
        for (uint32_t e = 0; e < nxp.experts_per_layer; ++e) {
            auto h = vmm->get_handle(l, e);
            REQUIRE(h != nos::INVALID_HANDLE);
            const uint8_t* data = vmm->pin(h);
            REQUIRE(data != nullptr);
            vmm->unpin(h);
        }
    }

    auto s1 = vmm->stats();
    CHECK(s1.total_pins == 8);
    CHECK(s1.cache_misses >= 4);  // At least first 4 are misses
    CHECK(s1.evictions >= 4);     // Need evictions to load experts 5-8
}

// ═══════════════════════════════════════════════════════════════════════════
// Statistics on repeated access (hit rate)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: repeated access improves hit rate", "[vmm_budget]") {
    // 1 layer * 4 experts = 4 experts, cache fits all
    TestNxp nxp(1, 4, 512, 16);

    nos::VmmFullConfig cfg{};
    cfg.nxp_path = nxp.path;
    cfg.user_budget_bytes = 512 * MB;  // Generous
    cfg.desired_context_length = 32;

    auto vmm = nos::Vmm::create(cfg);
    REQUIRE(vmm != nullptr);

    // Pin expert 0 once (cold miss)
    auto h0 = vmm->get_handle(0, 0);
    const uint8_t* d = vmm->pin(h0);
    REQUIRE(d != nullptr);
    vmm->unpin(h0);

    auto after_first = vmm->stats();
    CHECK(after_first.total_pins == 1);
    CHECK(after_first.cache_misses == 1);  // First access is a miss

    // Pin same expert 10 more times (should be hits)
    for (int i = 0; i < 10; ++i) {
        h0 = vmm->get_handle(0, 0);
        d = vmm->pin(h0);
        REQUIRE(d != nullptr);
        vmm->unpin(h0);
    }

    auto after_repeated = vmm->stats();
    CHECK(after_repeated.total_pins == 11);
    CHECK(after_repeated.cache_hits == 10);  // 10 of 11 are hits
    CHECK(after_repeated.hit_rate > 0.8);    // 10/11 ~ 0.909
}

// ═══════════════════════════════════════════════════════════════════════════
// Budget report format verification
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("VmmBudget: format_budget_report content", "[vmm_budget]") {
    // Sufficient
    nos::ModelParams p{};
    p.n_layers = 4;
    p.n_kv_heads = 8;
    p.head_dim = 128;
    p.hidden_dim = 256;
    p.max_expert_size = 1 * MB;
    p.top_k = 2;
    p.experts_per_layer = 8;

    auto bp_ok = nos::compute_budget(512 * MB, p, 512);
    REQUIRE(bp_ok.sufficient == true);
    std::string report_ok = nos::format_budget_report(bp_ok, p);
    CHECK_THAT(report_ok, ContainsSubstring("Expert cache:"));
    CHECK_THAT(report_ok, ContainsSubstring("KV cache:"));
    CHECK_THAT(report_ok, ContainsSubstring("Max context:"));
    CHECK_THAT(report_ok, ContainsSubstring("OK"));

    // Insufficient
    auto bp_bad = nos::compute_budget(1 * MB, p, 512);
    REQUIRE(bp_bad.sufficient == false);
    std::string report_bad = nos::format_budget_report(bp_bad, p);
    CHECK_THAT(report_bad, ContainsSubstring("INSUFFICIENT"));
    CHECK_THAT(report_bad, ContainsSubstring("Minimum needed:"));
}
