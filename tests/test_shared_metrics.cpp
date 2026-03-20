/// @file test_shared_metrics.cpp
/// @brief Catch2 tests for SharedMetrics, MetricsWriter, and MetricsReader.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "server/shared_metrics.h"

#include <chrono>
#include <cstring>
#include <thread>

#include <unistd.h>

using namespace nos;

TEST_CASE("SharedMetrics struct has expected magic value", "[shared_metrics]") {
    SharedMetrics m{};
    REQUIRE(m.magic == 0x4E4F534D);
}

TEST_CASE("SharedMetrics fits in a single page", "[shared_metrics]") {
    REQUIRE(sizeof(SharedMetrics) < 4096);
}

TEST_CASE("SharedMetrics is memcpy-safe", "[shared_metrics]") {
    SharedMetrics src{};
    src.tok_per_sec = 42.5;
    src.cache_hit_rate = 0.95;
    src.evictions = 100;
    src.resident_experts = 8;
    std::strncpy(src.prefetch_mode, "oracle", sizeof(src.prefetch_mode) - 1);

    SharedMetrics dst{};
    std::memcpy(&dst, &src, sizeof(SharedMetrics));

    REQUIRE(dst.magic == 0x4E4F534D);
    REQUIRE_THAT(dst.tok_per_sec, Catch::Matchers::WithinAbs(42.5, 0.01));
    REQUIRE_THAT(dst.cache_hit_rate, Catch::Matchers::WithinAbs(0.95, 0.001));
    REQUIRE(dst.evictions == 100);
    REQUIRE(dst.resident_experts == 8);
    REQUIRE(std::string(dst.prefetch_mode) == "oracle");
}

TEST_CASE("MetricsWriter creates shared memory segment", "[shared_metrics]") {
    std::string shm_name = "/nos_test_writer_" + std::to_string(getpid());

    {
        MetricsWriter writer(shm_name);
        REQUIRE(writer.is_open());
        REQUIRE(writer.shm_name() == shm_name);
    }
    // Destructor should shm_unlink
}

TEST_CASE("MetricsReader reads what MetricsWriter wrote", "[shared_metrics]") {
    std::string shm_name = "/nos_test_rw_" + std::to_string(getpid());

    MetricsWriter writer(shm_name);
    REQUIRE(writer.is_open());

    // Write metrics
    SharedMetrics src{};
    src.tok_per_sec = 123.4;
    src.ttft_ms = 50.0;
    src.cache_hit_rate = 0.87;
    src.evictions = 42;
    src.resident_experts = 16;
    src.oracle_rwp = 0.92;
    src.waste_ratio = 0.05;
    std::strncpy(src.prefetch_mode, "lstm", sizeof(src.prefetch_mode) - 1);
    src.switch_rate = 0.12;
    src.sticky_pct = 0.88;
    src.shift_detections = 3;
    src.active_slots = 2;
    src.max_slots = 4;
    src.last_update_epoch = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;

    writer.update(src);

    // Read metrics
    MetricsReader reader(shm_name);
    REQUIRE(reader.is_open());
    REQUIRE(reader.is_valid());

    SharedMetrics dst = reader.read();
    REQUIRE(dst.magic == 0x4E4F534D);
    REQUIRE_THAT(dst.tok_per_sec, Catch::Matchers::WithinAbs(123.4, 0.01));
    REQUIRE_THAT(dst.ttft_ms, Catch::Matchers::WithinAbs(50.0, 0.01));
    REQUIRE_THAT(dst.cache_hit_rate, Catch::Matchers::WithinAbs(0.87, 0.001));
    REQUIRE(dst.evictions == 42);
    REQUIRE(dst.resident_experts == 16);
    REQUIRE_THAT(dst.oracle_rwp, Catch::Matchers::WithinAbs(0.92, 0.001));
    REQUIRE(std::string(dst.prefetch_mode) == "lstm");
    REQUIRE_THAT(dst.switch_rate, Catch::Matchers::WithinAbs(0.12, 0.001));
    REQUIRE(dst.active_slots == 2);
    REQUIRE(dst.max_slots == 4);
}

TEST_CASE("Seqlock detects sequence mismatch", "[shared_metrics]") {
    // Verify that the sequence counter is even after a complete write
    std::string shm_name = "/nos_test_seq_" + std::to_string(getpid());

    MetricsWriter writer(shm_name);
    REQUIRE(writer.is_open());

    SharedMetrics m{};
    m.tok_per_sec = 10.0;
    m.last_update_epoch = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;

    // Write once
    writer.update(m);

    MetricsReader reader(shm_name);
    REQUIRE(reader.is_open());

    SharedMetrics result = reader.read();
    // Sequence should be even (2 after first write)
    REQUIRE((result.sequence & 1U) == 0);
    REQUIRE(result.sequence == 2);

    // Write again
    m.tok_per_sec = 20.0;
    writer.update(m);

    result = reader.read();
    REQUIRE((result.sequence & 1U) == 0);
    REQUIRE(result.sequence == 4);
    REQUIRE_THAT(result.tok_per_sec, Catch::Matchers::WithinAbs(20.0, 0.01));
}

TEST_CASE("MetricsReader::is_valid returns false for stale data", "[shared_metrics]") {
    std::string shm_name = "/nos_test_stale_" + std::to_string(getpid());

    MetricsWriter writer(shm_name);
    REQUIRE(writer.is_open());

    // Write with a very old timestamp
    SharedMetrics m{};
    m.last_update_epoch = 1000.0;  // Far in the past
    writer.update(m);

    MetricsReader reader(shm_name);
    REQUIRE(reader.is_open());
    REQUIRE_FALSE(reader.is_valid());
}

TEST_CASE("MetricsReader for non-existent segment", "[shared_metrics]") {
    MetricsReader reader("/nos_nonexistent_" + std::to_string(getpid()));
    REQUIRE_FALSE(reader.is_open());
    REQUIRE_FALSE(reader.is_valid());
}

TEST_CASE("Sparkline history round-trips through shared memory", "[shared_metrics]") {
    std::string shm_name = "/nos_test_spark_" + std::to_string(getpid());

    MetricsWriter writer(shm_name);
    REQUIRE(writer.is_open());

    SharedMetrics m{};
    m.last_update_epoch = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) / 1000.0;

    // Fill sparkline history
    for (size_t i = 0; i < 120; ++i) {
        m.tok_per_sec_history[i] = static_cast<float>(i) * 0.5f;
    }
    m.history_write_idx = 60;

    writer.update(m);

    MetricsReader reader(shm_name);
    SharedMetrics result = reader.read();

    REQUIRE(result.history_write_idx == 60);
    for (size_t i = 0; i < 120; ++i) {
        REQUIRE_THAT(static_cast<double>(result.tok_per_sec_history[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(i) * 0.5, 0.01));
    }
}
