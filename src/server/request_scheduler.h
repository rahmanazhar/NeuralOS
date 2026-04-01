#pragma once

/// @file request_scheduler.h
/// @brief Slot-based request scheduler for multi-sequence batching.
///
/// Each slot has its own nos_ctx_t with an independent KV cache,
/// enabling concurrent requests to share loaded experts via the VMM.
/// The OS page cache and VMM share expert data implicitly when all
/// contexts open the same .nxp file.

#include "api/libneuralos.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nos {

class RequestScheduler {
public:
    /// Aggregated metrics across all slots (for shared memory dashboard).
    struct AggregatedSlotMetrics {
        double tok_per_sec = 0.0;
        double ttft_ms = 0.0;
        double latency_p50_ms = 0.0;
        double latency_p95_ms = 0.0;
        double latency_p99_ms = 0.0;
        double cache_hit_rate = 0.0;
        uint64_t evictions = 0;
        uint32_t resident_experts = 0;
        double oracle_rwp = 0.0;
        double waste_ratio = 0.0;
        std::string prefetch_mode = "none";
        double switch_rate = 0.0;
        double sticky_pct = 0.0;
        uint32_t shift_detections = 0;
    };

    struct Slot {
        int slot_id = -1;
        nos_ctx_t* ctx = nullptr;
        bool active = false;
        std::mutex slot_mutex;
    };

    /// RAII guard that releases a slot when destroyed.
    class SlotGuard {
    public:
        SlotGuard(RequestScheduler& scheduler, Slot* slot)
            : scheduler_(scheduler), slot_(slot) {}

        ~SlotGuard() {
            if (slot_ != nullptr) {
                scheduler_.release_slot(slot_);
            }
        }

        SlotGuard(const SlotGuard&) = delete;
        SlotGuard& operator=(const SlotGuard&) = delete;

        SlotGuard(SlotGuard&& other) noexcept
            : scheduler_(other.scheduler_), slot_(other.slot_) {
            other.slot_ = nullptr;
        }

        SlotGuard& operator=(SlotGuard&&) = delete;

        Slot* get() const { return slot_; }
        Slot* operator->() const { return slot_; }
        explicit operator bool() const { return slot_ != nullptr; }

    private:
        RequestScheduler& scheduler_;
        Slot* slot_;
    };

    /// Construct scheduler.
    /// @param max_slots    Number of concurrent inference slots (default 4)
    /// @param model_path   Path to .nxp model directory
    /// @param base_config  Base configuration for each context
    RequestScheduler(size_t max_slots, const std::string& model_path,
                     nos_config_t base_config);
    ~RequestScheduler();

    RequestScheduler(const RequestScheduler&) = delete;
    RequestScheduler& operator=(const RequestScheduler&) = delete;

    /// Acquire an available slot. Returns nullptr if all slots busy.
    /// The slot's mutex is locked; call release_slot() when done.
    Slot* acquire_slot();

    /// Acquire a slot wrapped in an RAII guard.
    SlotGuard acquire_slot_guard();

    /// Release a slot: resets the context, marks inactive, unlocks mutex.
    void release_slot(Slot* slot);

    /// Number of currently active (in-use) slots.
    size_t active_count() const;

    /// Maximum number of slots.
    size_t slot_count() const;

    /// @return true if at least one slot was successfully created.
    bool is_ready() const;

    /// Aggregate metrics from all slots. Thread-safe (locks scheduler mutex).
    AggregatedSlotMetrics aggregate_metrics() const;

private:
    mutable std::mutex scheduler_mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
    size_t max_slots_;
    bool ready_ = false;
};

}  // namespace nos
