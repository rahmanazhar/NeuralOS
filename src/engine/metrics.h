#pragma once

/// @file metrics.h
/// @brief Lock-free metrics collection with atomic counters, histograms, and timeline.
///
/// Counters use atomic fetch_add for <5ns overhead. Histograms use
/// fixed pre-registered buckets with atomic counts. Timeline uses a
/// mutex (sampled infrequently, not on hot path).

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace nos {

class MetricsCollector {
public:
    MetricsCollector();

    // --- Atomic counters (lock-free after registration) ---
    void inc_counter(const std::string& name, uint64_t delta = 1);
    uint64_t get_counter(const std::string& name) const;

    // --- Fixed-bucket histogram (lock-free after registration) ---
    void register_histogram(const std::string& name, std::vector<double> bounds);
    void observe_histogram(const std::string& name, double value);

    struct HistogramSnapshot {
        std::vector<double> bucket_bounds;
        std::vector<uint64_t> bucket_counts;
        double sum = 0.0;
        uint64_t count = 0;
        double p50 = 0.0;
        double p95 = 0.0;
        double p99 = 0.0;
    };
    HistogramSnapshot get_histogram(const std::string& name) const;

    // --- Timeline (mutex-protected, sampled infrequently) ---
    void record_timeline(const std::string& name, double timestamp, double value);

    struct TimelineSnapshot {
        std::vector<std::pair<double, double>> points;
    };
    TimelineSnapshot get_timeline(const std::string& name) const;

    // --- Lifecycle ---
    void reset();
    nlohmann::json to_json() const;

    /// Register default metrics for inference engine.
    void register_defaults();

private:
    struct Histogram {
        std::vector<double> bounds;
        std::vector<std::unique_ptr<std::atomic<uint64_t>>> buckets;
        std::atomic<uint64_t> count{0};
        std::atomic<double> sum{0.0};
    };

    mutable std::mutex reg_mutex_;
    std::unordered_map<std::string, std::unique_ptr<std::atomic<uint64_t>>> counters_;
    std::unordered_map<std::string, Histogram> histograms_;

    mutable std::mutex timeline_mutex_;
    std::unordered_map<std::string, std::vector<std::pair<double, double>>> timelines_;

    std::atomic<uint64_t>& get_or_create_counter(const std::string& name);
};

}  // namespace nos
