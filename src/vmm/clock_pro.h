#pragma once

/// @file clock_pro.h
/// @brief CLOCK-Pro eviction policy with three-hand sweep algorithm.
///
/// Implements the CLOCK-Pro algorithm from USENIX ATC 2005 for
/// frequency-aware cache replacement. Uses an intrusive doubly-linked
/// circular list so that element removal preserves sweep ordering.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nos {

/// Node in the CLOCK-Pro circular doubly-linked list.
struct ClockProNode {
    uint32_t page_index;   ///< Index into VMM page table
    bool     referenced;   ///< Access bit (set on access, cleared by hand sweep)
    bool     hot;          ///< Hot (protected) or cold (eviction candidate)
    bool     in_test;      ///< Test period active
    bool     resident;     ///< Data in RAM (vs metadata-only non-resident cold)
    int      next;         ///< Index of next node in circular list (-1 = none)
    int      prev;         ///< Index of previous node in circular list (-1 = none)
    bool     active;       ///< Whether this pool slot is in use
};

/// For backward compatibility with tests using ClockProEntry
using ClockProEntry = ClockProNode;

class ClockPro {
public:
    /// @param max_resident    Maximum resident pages (= slab slot count)
    /// @param max_nonresident Maximum non-resident metadata entries (= 2 * max_resident)
    ClockPro(size_t max_resident, size_t max_nonresident);

    /// Insert a new page as cold resident. Evicts if at capacity.
    void insert(uint32_t page_index);

    /// Find and evict one cold resident page. Returns the evicted page_index.
    /// The evicted entry becomes non-resident (metadata retained for test period).
    uint32_t evict_one();

    /// Mark a page as accessed (set referenced bit).
    /// If the page is non-resident cold in test period, promotes to hot.
    void mark_accessed(uint32_t page_index);

    /// Remove a page entirely from the clock list.
    void remove(uint32_t page_index);

    /// Sweep hand_hot to demote unreferenced hot pages to cold.
    void run_hand_hot();

    /// Sweep hand_test to reclaim non-resident entries past their test period.
    void run_hand_test();

    /// Current count of resident pages in the clock list.
    size_t resident_count() const;

    /// Current count of non-resident pages in the clock list.
    size_t nonresident_count() const;

    /// Current cold target (adaptive parameter).
    size_t cold_target() const;

    /// Total entries in the clock list.
    size_t size() const;

private:
    size_t max_resident_;
    size_t max_nonresident_;
    size_t cold_target_;      ///< Target number of cold resident pages
    size_t resident_count_;
    size_t nonresident_count_;
    size_t hot_count_;
    size_t list_size_;        ///< Active nodes in circular list

    /// Node pool: pre-allocated to max_resident + max_nonresident.
    std::vector<ClockProNode> pool_;
    std::vector<int> free_slots_;  ///< Free pool slot indices

    /// Circular list head (-1 = empty list).
    int head_;

    /// Hand positions (indices into pool_, -1 = not set).
    int hand_cold_;
    int hand_hot_;
    int hand_test_;

    /// Map from page_index to pool slot index for O(1) lookup.
    std::unordered_map<uint32_t, int> index_map_;

    /// Allocate a node from the pool, returns pool index.
    int alloc_node();

    /// Free a node back to the pool.
    void free_node(int idx);

    /// Insert node into circular list (before head, i.e., at the "end").
    void link_node(int idx);

    /// Remove node from circular list (preserve ordering of other nodes).
    void unlink_node(int idx);

    /// Advance hand to next node in circular list.
    int advance_hand(int hand) const;
};

}  // namespace nos
