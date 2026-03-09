/// @file metrics.cpp
/// @brief Lock-free metrics collection implementation.

#include "engine/metrics.h"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

namespace nos {

MetricsCollector::MetricsCollector() = default;

std::atomic<uint64_t>& MetricsCollector::get_or_create_counter(const std::string& name) {
    {
        // Fast path: check if counter exists (no lock)
        auto it = counters_.find(name);
        if (it != counters_.end()) {
            return *it->second;
        }
    }
    // Slow path: create counter with lock
    std::lock_guard<std::mutex> lock(reg_mutex_);
    auto it = counters_.find(name);
    if (it != counters_.end()) {
        return *it->second;
    }
    auto [inserted, _] = counters_.emplace(
        name, std::make_unique<std::atomic<uint64_t>>(0));
    return *inserted->second;
}

void MetricsCollector::inc_counter(const std::string& name, uint64_t delta) {
    get_or_create_counter(name).fetch_add(delta, std::memory_order_relaxed);
}

uint64_t MetricsCollector::get_counter(const std::string& name) const {
    auto it = counters_.find(name);
    if (it == counters_.end()) return 0;
    return it->second->load(std::memory_order_relaxed);
}

void MetricsCollector::register_histogram(const std::string& name,
                                           std::vector<double> bounds) {
    std::lock_guard<std::mutex> lock(reg_mutex_);
    std::sort(bounds.begin(), bounds.end());

    // Erase existing (if any) and emplace fresh — atomics are non-assignable
    histograms_.erase(name);
    auto [it, _] = histograms_.emplace(std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple());
    auto& h = it->second;
    h.bounds = std::move(bounds);
    // One bucket per bound + one overflow bucket
    h.buckets.resize(h.bounds.size() + 1);
    for (auto& b : h.buckets) {
        b = std::make_unique<std::atomic<uint64_t>>(0);
    }
}

void MetricsCollector::observe_histogram(const std::string& name, double value) {
    auto it = histograms_.find(name);
    if (it == histograms_.end()) return;

    auto& h = it->second;

    // Binary search for bucket
    size_t idx = static_cast<size_t>(
        std::upper_bound(h.bounds.begin(), h.bounds.end(), value) - h.bounds.begin());
    h.buckets[idx]->fetch_add(1, std::memory_order_relaxed);
    h.count.fetch_add(1, std::memory_order_relaxed);

    // Atomic add for sum (CAS loop on double)
    double old_sum = h.sum.load(std::memory_order_relaxed);
    while (!h.sum.compare_exchange_weak(
               old_sum, old_sum + value, std::memory_order_relaxed)) {
        // CAS failed, old_sum updated automatically, retry
    }
}

MetricsCollector::HistogramSnapshot MetricsCollector::get_histogram(
        const std::string& name) const {
    HistogramSnapshot snap;
    auto it = histograms_.find(name);
    if (it == histograms_.end()) return snap;

    const auto& h = it->second;
    snap.bucket_bounds = h.bounds;
    snap.bucket_counts.resize(h.buckets.size());
    for (size_t i = 0; i < h.buckets.size(); i++) {
        snap.bucket_counts[i] = h.buckets[i]->load(std::memory_order_relaxed);
    }
    snap.sum = h.sum.load(std::memory_order_relaxed);
    snap.count = h.count.load(std::memory_order_relaxed);

    // Compute percentiles via linear interpolation from bucket counts
    if (snap.count > 0) {
        auto percentile = [&](double p) -> double {
            uint64_t target = static_cast<uint64_t>(
                std::ceil(p * static_cast<double>(snap.count)));
            if (target == 0) target = 1;
            uint64_t cumulative = 0;
            for (size_t i = 0; i < snap.bucket_counts.size(); i++) {
                cumulative += snap.bucket_counts[i];
                if (cumulative >= target) {
                    if (i < snap.bucket_bounds.size()) {
                        return snap.bucket_bounds[i];
                    }
                    // Overflow bucket — use last bound as estimate
                    return snap.bucket_bounds.empty() ? 0.0 : snap.bucket_bounds.back();
                }
            }
            return snap.bucket_bounds.empty() ? 0.0 : snap.bucket_bounds.back();
        };
        snap.p50 = percentile(0.50);
        snap.p95 = percentile(0.95);
        snap.p99 = percentile(0.99);
    }

    return snap;
}

void MetricsCollector::record_timeline(const std::string& name,
                                        double timestamp, double value) {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    timelines_[name].emplace_back(timestamp, value);
}

MetricsCollector::TimelineSnapshot MetricsCollector::get_timeline(
        const std::string& name) const {
    TimelineSnapshot snap;
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    auto it = timelines_.find(name);
    if (it != timelines_.end()) {
        snap.points = it->second;
    }
    return snap;
}

void MetricsCollector::reset() {
    std::lock_guard<std::mutex> lock(reg_mutex_);
    for (auto& [_, counter] : counters_) {
        counter->store(0, std::memory_order_relaxed);
    }
    for (auto& [_, hist] : histograms_) {
        for (auto& b : hist.buckets) {
            b->store(0, std::memory_order_relaxed);
        }
        hist.count.store(0, std::memory_order_relaxed);
        hist.sum.store(0.0, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> tl_lock(timeline_mutex_);
        timelines_.clear();
    }
}

nlohmann::json MetricsCollector::to_json() const {
    nlohmann::json j;

    // Counters
    nlohmann::json counters_j;
    for (const auto& [name, val] : counters_) {
        counters_j[name] = val->load(std::memory_order_relaxed);
    }
    j["counters"] = counters_j;

    // Histograms
    nlohmann::json hist_j;
    for (const auto& [name, _] : histograms_) {
        auto snap = get_histogram(name);
        nlohmann::json h;
        h["bounds"] = snap.bucket_bounds;
        h["counts"] = snap.bucket_counts;
        h["sum"] = snap.sum;
        h["count"] = snap.count;
        h["p50"] = snap.p50;
        h["p95"] = snap.p95;
        h["p99"] = snap.p99;
        hist_j[name] = h;
    }
    j["histograms"] = hist_j;

    // Timelines
    nlohmann::json tl_j;
    {
        std::lock_guard<std::mutex> lock(timeline_mutex_);
        for (const auto& [name, points] : timelines_) {
            nlohmann::json pts;
            for (const auto& [t, v] : points) {
                pts.push_back({{"t", t}, {"v", v}});
            }
            tl_j[name] = pts;
        }
    }
    j["timelines"] = tl_j;

    return j;
}

void MetricsCollector::register_defaults() {
    register_histogram("token_latency_ms",
        {1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000});
    register_histogram("io_latency_us",
        {10, 50, 100, 250, 500, 1000, 5000, 10000, 50000});
}

}  // namespace nos
