/// @file memory_budget.cpp
/// @brief Memory budget partitioning for the Virtual Memory Manager.
///
/// Deterministically partitions a user-specified --memory budget into:
///   expert_cache + kv_cache + working_buffers + os_overhead
/// Refuses to start if the budget is too small, reporting the minimum required.

#include "vmm/memory_budget.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

namespace nos {

namespace {

/// Round up to the next multiple of alignment.
size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// compute_budget
// ═══════════════════════════════════════════════════════════════════════════

BudgetPartition compute_budget(size_t user_budget_bytes,
                               const ModelParams& params,
                               uint32_t desired_context_length) {
    BudgetPartition bp{};
    bp.total = user_budget_bytes;
    bp.sufficient = true;
    bp.minimum_required = 0;

    // Step 1: OS overhead -- 512 MB conservative (runtime, stacks, .nxp index, code)
    bp.os_overhead = 512ULL * 1024 * 1024;

    // Step 2: Working buffers -- input + output + scratch activations, doubled for safety
    bp.working_buffers = align_up(
        static_cast<size_t>(params.hidden_dim) * sizeof(float) * 3 * 2, 64);

    // Step 3: KV cache sizing
    // kv_per_token = 2 * n_layers * n_kv_heads * head_dim * sizeof(uint16_t)
    // 2 for K+V, FP16 storage
    size_t kv_per_token = 2ULL * params.n_layers * params.n_kv_heads
                          * params.head_dim * sizeof(uint16_t);
    bp.kv_cache = kv_per_token * desired_context_length;

    // Step 4: Check if reserved exceeds budget
    size_t reserved = bp.os_overhead + bp.working_buffers + bp.kv_cache;
    if (reserved >= user_budget_bytes) {
        bp.sufficient = false;
        // Minimum required: overhead + working + kv for 512 tokens + min expert cache
        uint32_t min_expert_slots = params.top_k * 2;
        bp.minimum_required = bp.os_overhead + bp.working_buffers
                              + kv_per_token * 512
                              + static_cast<size_t>(min_expert_slots) * params.max_expert_size;
        bp.expert_cache = 0;
        bp.expert_slots = 0;
        bp.max_context = 0;
        return bp;
    }

    // Step 5: Expert cache = remaining after reserved
    bp.expert_cache = user_budget_bytes - reserved;

    // Step 6: Expert slots
    bp.expert_slots = static_cast<uint32_t>(bp.expert_cache / params.max_expert_size);

    // Step 7: Minimum expert slot check
    uint32_t min_expert_slots = params.top_k * 2;
    if (bp.expert_slots < min_expert_slots) {
        bp.sufficient = false;
        bp.minimum_required = bp.os_overhead + bp.working_buffers
                              + kv_per_token * 512
                              + static_cast<size_t>(min_expert_slots) * params.max_expert_size;
        return bp;
    }

    // Step 8: Max context -- computed with minimum expert cache
    size_t min_expert_cache = static_cast<size_t>(min_expert_slots) * params.max_expert_size;
    size_t available_for_kv = user_budget_bytes - bp.os_overhead
                              - bp.working_buffers - min_expert_cache;
    bp.max_context = static_cast<uint32_t>(available_for_kv / kv_per_token);

    return bp;
}

// ═══════════════════════════════════════════════════════════════════════════
// format_bytes
// ═══════════════════════════════════════════════════════════════════════════

std::string format_bytes(size_t bytes) {
    char buf[64];
    if (bytes >= (1ULL << 30)) {
        std::snprintf(buf, sizeof(buf), "%.2f GB",
                      static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f MB",
                      static_cast<double>(bytes) / static_cast<double>(1ULL << 20));
    }
    return buf;
}

// ═══════════════════════════════════════════════════════════════════════════
// format_budget_report
// ═══════════════════════════════════════════════════════════════════════════

std::string format_budget_report(const BudgetPartition& bp,
                                 const ModelParams& params) {
    std::ostringstream os;

    if (!bp.sufficient) {
        os << "Memory Budget: INSUFFICIENT\n"
           << "  Requested:       " << format_bytes(bp.total) << "\n"
           << "  Minimum needed:  " << format_bytes(bp.minimum_required) << "\n"
           << "  Breakdown of minimum:\n";

        uint32_t min_expert_slots = params.top_k * 2;
        size_t min_expert_cache = static_cast<size_t>(min_expert_slots) * params.max_expert_size;
        size_t kv_per_token = 2ULL * params.n_layers * params.n_kv_heads
                              * params.head_dim * sizeof(uint16_t);
        size_t min_kv = kv_per_token * 512;

        os << "    Expert cache:    " << format_bytes(min_expert_cache)
           << "  (" << min_expert_slots << " slots minimum for top_k="
           << params.top_k << ")\n"
           << "    KV cache:        " << format_bytes(min_kv)
           << "  (512 token minimum)\n"
           << "    Working buffers: " << format_bytes(bp.working_buffers) << "\n"
           << "    OS overhead:     " << format_bytes(bp.os_overhead) << "\n";

        return os.str();
    }

    // Sufficient budget report
    os << "Memory Budget:\n"
       << "  Total:           " << format_bytes(bp.total) << "\n"
       << "  Expert cache:    " << format_bytes(bp.expert_cache)
       << "  (" << bp.expert_slots << " slots x "
       << format_bytes(params.max_expert_size) << ")\n"
       << "  KV cache:        " << format_bytes(bp.kv_cache)
       << "  (max context: " << bp.max_context << " tokens)\n"
       << "  Working buffers: " << format_bytes(bp.working_buffers) << "\n"
       << "  OS overhead:     " << format_bytes(bp.os_overhead) << "\n"
       << "  ---\n"
       << "  Status: OK\n"
       << "  Max context:     " << bp.max_context << " tokens\n";

    if (bp.max_context < 2048) {
        os << "  WARNING: Max context (" << bp.max_context
           << " tokens) is below 2048. Consider increasing --memory.\n";
    }

    return os.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// parse_memory_string
// ═══════════════════════════════════════════════════════════════════════════

size_t parse_memory_string(const std::string& str) {
    if (str.empty()) return 0;

    // Find where the numeric part ends and suffix begins
    size_t pos = 0;
    while (pos < str.size() && (std::isdigit(str[pos]) || str[pos] == '.')) {
        ++pos;
    }

    if (pos == 0) return 0;  // No numeric prefix

    // Parse numeric value
    double value = 0;
    try {
        value = std::stod(str.substr(0, pos));
    } catch (...) {
        return 0;
    }

    if (value <= 0) return 0;

    // Parse suffix (case-insensitive)
    std::string suffix = str.substr(pos);
    // Convert to uppercase
    for (auto& c : suffix) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    size_t multiplier = 0;
    if (suffix == "G" || suffix == "GB") {
        multiplier = 1ULL << 30;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1ULL << 20;
    } else if (suffix.empty()) {
        // Raw bytes
        multiplier = 1;
    } else {
        return 0;  // Unknown suffix
    }

    return static_cast<size_t>(value * static_cast<double>(multiplier));
}

}  // namespace nos
