/// @file clock_pro.cpp
/// @brief CLOCK-Pro three-hand eviction algorithm implementation.
///
/// Based on the USENIX ATC 2005 paper "CLOCK-Pro: An Effective Improvement
/// of the CLOCK Replacement" and the RobinSystems C reference implementation.
///
/// Uses an intrusive doubly-linked circular list (stored in a node pool)
/// so that node removal preserves the sweep ordering of the three hands.

#include "clock_pro.h"

#include <cassert>
#include <limits>

namespace nos {

ClockPro::ClockPro(size_t max_resident, size_t max_nonresident)
    : max_resident_(max_resident)
    , max_nonresident_(max_nonresident)
    , cold_target_(max_resident / 2)
    , resident_count_(0)
    , nonresident_count_(0)
    , hot_count_(0)
    , list_size_(0)
    , head_(-1)
    , hand_cold_(-1)
    , hand_hot_(-1)
    , hand_test_(-1) {
    // Pre-allocate pool
    size_t pool_size = max_resident + max_nonresident + 16;  // +16 headroom
    pool_.resize(pool_size);
    for (auto& node : pool_) {
        node.active = false;
        node.next = -1;
        node.prev = -1;
    }
    free_slots_.reserve(pool_size);
    for (int i = static_cast<int>(pool_size) - 1; i >= 0; --i) {
        free_slots_.push_back(i);
    }
}

int ClockPro::alloc_node() {
    if (free_slots_.empty()) return -1;
    int idx = free_slots_.back();
    free_slots_.pop_back();
    pool_[static_cast<size_t>(idx)].active = true;
    pool_[static_cast<size_t>(idx)].next = -1;
    pool_[static_cast<size_t>(idx)].prev = -1;
    return idx;
}

void ClockPro::free_node(int idx) {
    assert(idx >= 0 && static_cast<size_t>(idx) < pool_.size());
    pool_[static_cast<size_t>(idx)].active = false;
    free_slots_.push_back(idx);
}

void ClockPro::link_node(int idx) {
    assert(idx >= 0);
    auto& node = pool_[static_cast<size_t>(idx)];

    if (head_ == -1) {
        // Empty list: self-loop
        node.next = idx;
        node.prev = idx;
        head_ = idx;
    } else {
        // Insert before head (at the "tail" of the circular list)
        auto& head_node = pool_[static_cast<size_t>(head_)];
        int tail = head_node.prev;
        auto& tail_node = pool_[static_cast<size_t>(tail)];

        node.next = head_;
        node.prev = tail;
        tail_node.next = idx;
        head_node.prev = idx;
    }
    ++list_size_;
}

void ClockPro::unlink_node(int idx) {
    assert(idx >= 0);
    auto& node = pool_[static_cast<size_t>(idx)];

    if (list_size_ == 1) {
        // Last node
        head_ = -1;
        hand_cold_ = -1;
        hand_hot_ = -1;
        hand_test_ = -1;
    } else {
        int p = node.prev;
        int n = node.next;
        pool_[static_cast<size_t>(p)].next = n;
        pool_[static_cast<size_t>(n)].prev = p;

        // Fix hands if they point to the removed node
        if (hand_cold_ == idx) hand_cold_ = n;
        if (hand_hot_ == idx) hand_hot_ = n;
        if (hand_test_ == idx) hand_test_ = n;
        if (head_ == idx) head_ = n;
    }

    node.next = -1;
    node.prev = -1;
    --list_size_;
}

int ClockPro::advance_hand(int hand) const {
    if (hand == -1 || list_size_ == 0) return -1;
    return pool_[static_cast<size_t>(hand)].next;
}

void ClockPro::insert(uint32_t page_index) {
    // Check if page already exists (non-resident metadata or resident)
    auto it = index_map_.find(page_index);
    if (it != index_map_.end()) {
        int idx = it->second;
        auto& node = pool_[static_cast<size_t>(idx)];

        if (!node.resident) {
            // Re-reference of non-resident page
            if (node.in_test) {
                // Test-period hit: promote to hot
                node.hot = true;
                node.in_test = false;
                if (nonresident_count_ > 0) --nonresident_count_;
                ++hot_count_;
                if (cold_target_ < max_resident_) {
                    ++cold_target_;
                }
            } else {
                // Non-resident, test expired: treat as new cold
                if (nonresident_count_ > 0) --nonresident_count_;
            }
            node.resident = true;
            node.referenced = false;
            ++resident_count_;

            if (resident_count_ > max_resident_) {
                evict_one();
            }
            return;
        }
        // Already resident: mark accessed
        mark_accessed(page_index);
        return;
    }

    // Evict if at capacity
    if (resident_count_ >= max_resident_) {
        evict_one();
    }

    // Allocate new node
    int idx = alloc_node();
    if (idx == -1) {
        // Pool exhausted: reclaim oldest non-resident entries
        run_hand_test();
        idx = alloc_node();
        if (idx == -1) return;  // Cannot allocate
    }

    auto& node = pool_[static_cast<size_t>(idx)];
    node.page_index = page_index;
    node.referenced = false;
    node.hot = false;
    node.in_test = true;   // New cold pages start in test period
    node.resident = true;

    link_node(idx);
    index_map_[page_index] = idx;
    ++resident_count_;

    // Initialize hands if needed
    if (hand_cold_ == -1) hand_cold_ = idx;
    if (hand_hot_ == -1) hand_hot_ = idx;
    if (hand_test_ == -1) hand_test_ = idx;

    // Clean up non-resident if too many
    if (nonresident_count_ > max_nonresident_) {
        run_hand_test();
    }
}

uint32_t ClockPro::evict_one() {
    if (list_size_ == 0 || resident_count_ == 0) {
        return std::numeric_limits<uint32_t>::max();
    }

    if (hand_cold_ == -1) hand_cold_ = head_;

    // Limit iterations to prevent infinite loops
    size_t max_iterations = list_size_ * 3;
    size_t iterations = 0;

    while (iterations < max_iterations) {
        if (hand_cold_ == -1) break;
        auto& node = pool_[static_cast<size_t>(hand_cold_)];
        ++iterations;

        // Skip non-resident entries
        if (!node.resident) {
            hand_cold_ = advance_hand(hand_cold_);
            continue;
        }

        // Skip hot entries
        if (node.hot) {
            hand_cold_ = advance_hand(hand_cold_);
            continue;
        }

        // Cold resident page
        if (node.referenced) {
            // Referenced cold page in test period: promote to hot
            if (node.in_test) {
                node.hot = true;
                node.in_test = false;
                node.referenced = false;
                ++hot_count_;
                if (cold_target_ < max_resident_) {
                    ++cold_target_;
                }
            } else {
                // Give another chance
                node.referenced = false;
            }
            hand_cold_ = advance_hand(hand_cold_);
            continue;
        }

        // Unreferenced cold resident page -- EVICT
        uint32_t evicted_page = node.page_index;

        // Transition to non-resident (retain metadata for test period)
        node.resident = false;
        node.in_test = true;
        node.referenced = false;
        if (resident_count_ > 0) --resident_count_;
        ++nonresident_count_;

        hand_cold_ = advance_hand(hand_cold_);

        // Run hand_hot if cold resident count drops below target
        size_t cold_resident = resident_count_ > hot_count_
                               ? resident_count_ - hot_count_ : 0;
        if (cold_resident < cold_target_ && hot_count_ > 0) {
            run_hand_hot();
        }

        // Clean up non-resident if too many
        if (nonresident_count_ > max_nonresident_) {
            run_hand_test();
        }

        return evicted_page;
    }

    // Fallback: force-evict first cold resident found
    if (head_ != -1) {
        int cur = head_;
        for (size_t i = 0; i < list_size_; ++i) {
            auto& node = pool_[static_cast<size_t>(cur)];
            if (node.resident && !node.hot) {
                uint32_t evicted_page = node.page_index;
                node.resident = false;
                node.in_test = true;
                node.referenced = false;
                if (resident_count_ > 0) --resident_count_;
                ++nonresident_count_;
                return evicted_page;
            }
            cur = node.next;
        }
    }

    // Ultra-fallback: demote a hot page and evict
    if (head_ != -1) {
        int cur = head_;
        for (size_t i = 0; i < list_size_; ++i) {
            auto& node = pool_[static_cast<size_t>(cur)];
            if (node.resident && node.hot) {
                node.hot = false;
                if (hot_count_ > 0) --hot_count_;
                uint32_t evicted_page = node.page_index;
                node.resident = false;
                node.in_test = true;
                node.referenced = false;
                if (resident_count_ > 0) --resident_count_;
                ++nonresident_count_;
                return evicted_page;
            }
            cur = node.next;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

void ClockPro::mark_accessed(uint32_t page_index) {
    auto it = index_map_.find(page_index);
    if (it == index_map_.end()) return;

    int idx = it->second;
    auto& node = pool_[static_cast<size_t>(idx)];

    if (!node.resident) {
        // Non-resident access during test period: promote to hot
        if (node.in_test) {
            node.resident = true;
            node.hot = true;
            node.in_test = false;
            node.referenced = false;
            if (nonresident_count_ > 0) --nonresident_count_;
            ++resident_count_;
            ++hot_count_;

            if (cold_target_ < max_resident_) {
                ++cold_target_;
            }

            if (resident_count_ > max_resident_) {
                evict_one();
            }
        }
        return;
    }

    // Resident cold page in test period: promote to hot immediately
    if (!node.hot && node.in_test) {
        node.hot = true;
        node.in_test = false;
        node.referenced = false;
        ++hot_count_;
        if (cold_target_ < max_resident_) {
            ++cold_target_;
        }
        return;
    }

    // Set referenced bit
    node.referenced = true;
}

void ClockPro::remove(uint32_t page_index) {
    auto it = index_map_.find(page_index);
    if (it == index_map_.end()) return;

    int idx = it->second;
    auto& node = pool_[static_cast<size_t>(idx)];

    // Update counters
    if (node.resident) {
        if (resident_count_ > 0) --resident_count_;
        if (node.hot && hot_count_ > 0) --hot_count_;
    } else {
        if (nonresident_count_ > 0) --nonresident_count_;
    }

    index_map_.erase(it);
    unlink_node(idx);
    free_node(idx);
}

void ClockPro::run_hand_hot() {
    if (list_size_ == 0 || hot_count_ == 0) return;
    if (hand_hot_ == -1) hand_hot_ = head_;

    size_t max_iterations = list_size_ * 2;
    size_t iterations = 0;

    while (iterations < max_iterations && hot_count_ > 0) {
        if (hand_hot_ == -1) break;
        auto& node = pool_[static_cast<size_t>(hand_hot_)];
        ++iterations;

        if (!node.hot || !node.resident) {
            hand_hot_ = advance_hand(hand_hot_);
            continue;
        }

        if (node.referenced) {
            node.referenced = false;
            hand_hot_ = advance_hand(hand_hot_);
            continue;
        }

        // Hot and unreferenced: demote to cold
        node.hot = false;
        if (hot_count_ > 0) --hot_count_;
        hand_hot_ = advance_hand(hand_hot_);
        return;
    }
}

void ClockPro::run_hand_test() {
    if (list_size_ == 0 || nonresident_count_ == 0) return;
    if (hand_test_ == -1) hand_test_ = head_;

    size_t max_iterations = list_size_ * 2;
    size_t iterations = 0;

    while (iterations < max_iterations && nonresident_count_ > max_nonresident_) {
        if (hand_test_ == -1) break;

        int cur = hand_test_;
        auto& node = pool_[static_cast<size_t>(cur)];
        ++iterations;

        if (node.resident) {
            hand_test_ = advance_hand(hand_test_);
            continue;
        }

        // Non-resident entry: remove and reclaim
        if (node.in_test && cold_target_ > 1) {
            --cold_target_;
        }

        // Advance hand before removal
        hand_test_ = advance_hand(hand_test_);

        // Remove the node entirely
        if (nonresident_count_ > 0) --nonresident_count_;
        index_map_.erase(node.page_index);
        unlink_node(cur);
        free_node(cur);
    }
}

size_t ClockPro::resident_count() const {
    return resident_count_;
}

size_t ClockPro::nonresident_count() const {
    return nonresident_count_;
}

size_t ClockPro::cold_target() const {
    return cold_target_;
}

size_t ClockPro::size() const {
    return list_size_;
}

}  // namespace nos
