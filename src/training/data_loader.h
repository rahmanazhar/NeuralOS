#pragma once

/// @file data_loader.h
/// @brief JSONL data loading and mini-batch creation for training.
///
/// Reads JSONL files where each line has a "text" field, tokenizes
/// using the NeuralOS Tokenizer, and produces mini-batches with
/// cursor-based epoch wrapping.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nos {

class Tokenizer;

class DataLoader {
public:
    /// A mini-batch of tokenized sequences.
    struct Batch {
        std::vector<std::vector<int>> token_sequences;
        size_t total_tokens = 0;
    };

    DataLoader() = default;

    /// Load a JSONL file. Each line must be a JSON object with a "text" field.
    /// Lines without "text" are skipped. Returns true if at least one text loaded.
    bool load(const std::string& jsonl_path);

    /// Create next mini-batch by tokenizing batch_size texts.
    /// Truncates sequences to max_seq_len. Cursor wraps on epoch boundary.
    Batch next_batch(const Tokenizer& tokenizer, size_t batch_size,
                     size_t max_seq_len);

    /// Shuffle the text entries with a given seed.
    void shuffle(uint64_t seed);

    /// Number of text entries loaded.
    size_t size() const;

    /// True if no text entries loaded.
    bool empty() const;

private:
    std::vector<std::string> texts_;
    size_t cursor_ = 0;
};

}  // namespace nos
