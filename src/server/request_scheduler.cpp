/// @file request_scheduler.cpp
/// @brief Slot-based request scheduler implementation.

#include "server/request_scheduler.h"

#include <cstdio>

#include <nlohmann/json.hpp>

namespace nos {

RequestScheduler::RequestScheduler(size_t max_slots,
                                   const std::string& model_path,
                                   nos_config_t base_config)
    : max_slots_(max_slots)
{
    slots_.reserve(max_slots_);

    for (size_t i = 0; i < max_slots_; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->slot_id = static_cast<int>(i);

        // Each slot gets its own context with independent KV cache.
        // All contexts share the same model file, so the VMM/OS page
        // cache shares expert data implicitly.
        slot->ctx = nos_create(model_path.c_str(), base_config);
        if (slot->ctx == nullptr) {
            std::fprintf(stderr,
                "RequestScheduler: Failed to create context for slot %zu: %s\n",
                i, nos_last_error());
            // Continue -- some slots may still work
        }

        slot->active = false;
        slots_.push_back(std::move(slot));
    }

    // Check if at least one slot is usable
    for (const auto& slot : slots_) {
        if (slot->ctx != nullptr) {
            ready_ = true;
            break;
        }
    }
}

RequestScheduler::~RequestScheduler() {
    for (auto& slot : slots_) {
        if (slot->ctx != nullptr) {
            nos_destroy(slot->ctx);
            slot->ctx = nullptr;
        }
    }
}

RequestScheduler::Slot* RequestScheduler::acquire_slot() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);

    for (auto& slot : slots_) {
        if (!slot->active && slot->ctx != nullptr) {
            slot->active = true;
            slot->slot_mutex.lock();
            return slot.get();
        }
    }

    return nullptr;  // All slots busy
}

RequestScheduler::SlotGuard RequestScheduler::acquire_slot_guard() {
    return SlotGuard(*this, acquire_slot());
}

void RequestScheduler::release_slot(Slot* slot) {
    if (slot == nullptr) return;

    // Reset KV cache for next request
    if (slot->ctx != nullptr) {
        nos_reset(slot->ctx);
    }

    slot->slot_mutex.unlock();

    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    slot->active = false;
}

size_t RequestScheduler::active_count() const {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    size_t count = 0;
    for (const auto& slot : slots_) {
        if (slot->active) ++count;
    }
    return count;
}

size_t RequestScheduler::slot_count() const {
    return max_slots_;
}

bool RequestScheduler::is_ready() const {
    return ready_;
}

RequestScheduler::AggregatedSlotMetrics RequestScheduler::aggregate_metrics() const {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    AggregatedSlotMetrics agg{};

    double ttft_sum = 0.0;
    int ttft_count = 0;

    for (const auto& slot : slots_) {
        if (slot->ctx == nullptr) continue;

        const char* json_str = nos_get_metrics(slot->ctx);
        if (json_str == nullptr) continue;

        try {
            auto j = nlohmann::json::parse(json_str);

            // TTFT
            double ttft = j.value("ttft_ms", 0.0);
            if (ttft > 0.0) {
                ttft_sum += ttft;
                ++ttft_count;
            }

            // Histograms: token_latency_ms percentiles and tok/s estimation
            if (j.contains("histograms") && j["histograms"].contains("token_latency_ms")) {
                const auto& h = j["histograms"]["token_latency_ms"];
                double p50 = h.value("p50", 0.0);
                double p95 = h.value("p95", 0.0);
                double p99 = h.value("p99", 0.0);
                // Take worst-case (max) across slots
                if (p50 > agg.latency_p50_ms) agg.latency_p50_ms = p50;
                if (p95 > agg.latency_p95_ms) agg.latency_p95_ms = p95;
                if (p99 > agg.latency_p99_ms) agg.latency_p99_ms = p99;
                // Estimate tok/s from histogram: count / (sum / 1000)
                uint64_t count = h.value("count", static_cast<uint64_t>(0));
                double sum_ms = h.value("sum", 0.0);
                if (sum_ms > 0.0) {
                    agg.tok_per_sec += static_cast<double>(count) / (sum_ms / 1000.0);
                }
            }

            // Routing
            if (j.contains("routing")) {
                agg.switch_rate = j["routing"].value("switch_rate", 0.0);
                double wl = j["routing"].value("avg_window_length", 0.0);
                if (wl > 0.0) {
                    agg.sticky_pct = 1.0 - agg.switch_rate;
                }
                agg.shift_detections = static_cast<uint32_t>(
                    j["routing"].value("total_switches", static_cast<uint64_t>(0)));
            }

            // Prefetch
            if (j.contains("prefetch")) {
                agg.prefetch_mode = j["prefetch"].value("mode", std::string("none"));
                agg.oracle_rwp = j["prefetch"].value("rwp_oracle", 0.0);
                uint64_t hits = j["prefetch"].value("speculative_hits",
                                                    static_cast<uint64_t>(0));
                uint64_t misses = j["prefetch"].value("speculative_misses",
                                                      static_cast<uint64_t>(0));
                if (hits + misses > 0) {
                    agg.waste_ratio = static_cast<double>(misses)
                        / static_cast<double>(hits + misses);
                }
            }

            // VMM
            if (j.contains("vmm")) {
                agg.cache_hit_rate = j["vmm"].value("hit_rate", 0.0);
                agg.evictions = j["vmm"].value("evictions", static_cast<uint64_t>(0));
                agg.resident_experts = static_cast<uint32_t>(
                    j["vmm"].value("resident_pages", static_cast<uint32_t>(0)));
            }

        } catch (...) {
            // Skip malformed JSON
            continue;
        }
    }

    // Average TTFT across slots with data
    if (ttft_count > 0) {
        agg.ttft_ms = ttft_sum / static_cast<double>(ttft_count);
    }

    return agg;
}

}  // namespace nos
