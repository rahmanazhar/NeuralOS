/// @file test_metrics.cpp
/// @brief Tests for MetricsCollector (counters, histograms, timeline, JSON).

#include "engine/metrics.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nlohmann/json.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Counter increment and get", "[metrics]") {
    nos::MetricsCollector mc;
    mc.inc_counter("tokens");
    REQUIRE(mc.get_counter("tokens") == 1);
}

TEST_CASE("Counter increment is additive", "[metrics]") {
    nos::MetricsCollector mc;
    mc.inc_counter("tokens", 5);
    mc.inc_counter("tokens", 3);
    REQUIRE(mc.get_counter("tokens") == 8);
}

TEST_CASE("Counter get returns 0 for unknown", "[metrics]") {
    nos::MetricsCollector mc;
    REQUIRE(mc.get_counter("nonexistent") == 0);
}

TEST_CASE("Histogram observe distributes to correct buckets", "[metrics]") {
    nos::MetricsCollector mc;
    mc.register_histogram("latency", {10.0, 50.0, 100.0});

    mc.observe_histogram("latency", 5.0);    // bucket 0: [0, 10)
    mc.observe_histogram("latency", 25.0);   // bucket 1: [10, 50)
    mc.observe_histogram("latency", 75.0);   // bucket 2: [50, 100)
    mc.observe_histogram("latency", 200.0);  // bucket 3: overflow [100, inf)

    auto snap = mc.get_histogram("latency");
    REQUIRE(snap.bucket_counts.size() == 4);
    REQUIRE(snap.bucket_counts[0] == 1);
    REQUIRE(snap.bucket_counts[1] == 1);
    REQUIRE(snap.bucket_counts[2] == 1);
    REQUIRE(snap.bucket_counts[3] == 1);
    REQUIRE(snap.count == 4);
    REQUIRE_THAT(snap.sum, WithinAbs(305.0, 0.01));
}

TEST_CASE("Histogram p50/p95/p99 for known distribution", "[metrics]") {
    nos::MetricsCollector mc;
    mc.register_histogram("lat", {10.0, 50.0, 100.0, 500.0});

    // 100 samples: 60 in [0,10), 30 in [10,50), 8 in [50,100), 2 in [100,500)
    for (int i = 0; i < 60; i++) mc.observe_histogram("lat", 5.0);
    for (int i = 0; i < 30; i++) mc.observe_histogram("lat", 25.0);
    for (int i = 0; i < 8; i++)  mc.observe_histogram("lat", 75.0);
    for (int i = 0; i < 2; i++)  mc.observe_histogram("lat", 200.0);

    auto snap = mc.get_histogram("lat");
    REQUIRE(snap.count == 100);

    // p50: target=50, cumulative hits 60 at bucket 0 → bound 10.0
    REQUIRE_THAT(snap.p50, WithinAbs(10.0, 0.01));
    // p95: target=95, cumulative 60+30=90 at bucket 1, 60+30+8=98 at bucket 2 → bound 100.0
    REQUIRE_THAT(snap.p95, WithinAbs(100.0, 0.01));
    // p99: target=99, cumulative 98 at bucket 2 → need bucket 3 → bound 500.0
    REQUIRE_THAT(snap.p99, WithinAbs(500.0, 0.01));
}

TEST_CASE("Timeline recording stores points", "[metrics]") {
    nos::MetricsCollector mc;
    mc.record_timeline("io_pressure", 1.0, 0.5);
    mc.record_timeline("io_pressure", 2.0, 0.7);

    auto snap = mc.get_timeline("io_pressure");
    REQUIRE(snap.points.size() == 2);
    REQUIRE_THAT(snap.points[0].first, WithinAbs(1.0, 0.001));
    REQUIRE_THAT(snap.points[0].second, WithinAbs(0.5, 0.001));
    REQUIRE_THAT(snap.points[1].first, WithinAbs(2.0, 0.001));
    REQUIRE_THAT(snap.points[1].second, WithinAbs(0.7, 0.001));
}

TEST_CASE("to_json produces valid JSON with counters and histograms", "[metrics]") {
    nos::MetricsCollector mc;
    mc.inc_counter("tokens", 42);
    mc.register_histogram("lat", {10.0, 100.0});
    mc.observe_histogram("lat", 5.0);
    mc.record_timeline("pressure", 0.0, 1.0);

    auto j = mc.to_json();
    REQUIRE(j.contains("counters"));
    REQUIRE(j["counters"]["tokens"] == 42);
    REQUIRE(j.contains("histograms"));
    REQUIRE(j["histograms"].contains("lat"));
    REQUIRE(j["histograms"]["lat"]["count"] == 1);
    REQUIRE(j.contains("timelines"));
    REQUIRE(j["timelines"].contains("pressure"));
}

TEST_CASE("reset zeros counters and histogram buckets", "[metrics]") {
    nos::MetricsCollector mc;
    mc.inc_counter("tokens", 10);
    mc.register_histogram("lat", {10.0, 100.0});
    mc.observe_histogram("lat", 5.0);
    mc.record_timeline("pressure", 0.0, 1.0);

    mc.reset();

    REQUIRE(mc.get_counter("tokens") == 0);
    auto snap = mc.get_histogram("lat");
    REQUIRE(snap.count == 0);
    REQUIRE_THAT(snap.sum, WithinAbs(0.0, 0.001));
    auto tl = mc.get_timeline("pressure");
    REQUIRE(tl.points.empty());
}

TEST_CASE("register_defaults creates expected histograms", "[metrics]") {
    nos::MetricsCollector mc;
    mc.register_defaults();

    // Should have these histograms registered
    auto tok_lat = mc.get_histogram("token_latency_ms");
    REQUIRE(tok_lat.bucket_bounds.size() == 10);
    auto io_lat = mc.get_histogram("io_latency_us");
    REQUIRE(io_lat.bucket_bounds.size() == 9);
}
