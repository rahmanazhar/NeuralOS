/// @file memory_budget.cpp
/// @brief STUB -- memory budget partitioning (to be implemented in GREEN phase).

#include "vmm/memory_budget.h"

namespace nos {

BudgetPartition compute_budget(size_t /*user_budget_bytes*/,
                               const ModelParams& /*params*/,
                               uint32_t /*desired_context_length*/) {
    BudgetPartition bp{};
    return bp;  // All zeros -- tests should fail
}

std::string format_budget_report(const BudgetPartition& /*bp*/,
                                 const ModelParams& /*params*/) {
    return "";  // Stub
}

std::string format_bytes(size_t /*bytes*/) {
    return "";  // Stub
}

size_t parse_memory_string(const std::string& /*str*/) {
    return 0;  // Stub
}

}  // namespace nos
