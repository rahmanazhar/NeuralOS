/// @file request_scheduler.cpp
/// @brief Slot-based request scheduler implementation.

#include "server/request_scheduler.h"

#include <cstdio>

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

}  // namespace nos
