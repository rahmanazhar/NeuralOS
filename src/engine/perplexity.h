#pragma once

/// @file perplexity.h
/// @brief Perplexity computation utility for model quality validation.
///
/// Follows llama.cpp methodology: non-overlapping context-chunk evaluation.
/// Used by the CLI for the perplexity budget gate (PIPE-12) and the
/// `neuralos perplexity` subcommand.

#include <vector>

namespace nos {

class InferenceEngine;

/// Compute perplexity on a token sequence using non-overlapping context chunks.
///
/// For each chunk, runs forward pass, computes cross-entropy loss on
/// next-token predictions. Returns perplexity = exp(mean_loss).
///
/// @param engine          Initialized inference engine
/// @param tokens          Token sequence to evaluate
/// @param context_length  Chunk size (0 = use model's max_seq_len)
/// @return Perplexity value (lower is better), or -1.0 on error
double compute_perplexity(InferenceEngine& engine,
                          const std::vector<int>& tokens,
                          int context_length = 0);

}  // namespace nos
