/// @file data_loader.cpp
/// @brief JSONL data loader implementation.

#include "training/data_loader.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

#include <nlohmann/json.hpp>

namespace nos {

bool DataLoader::load(const std::string& jsonl_path) {
    std::ifstream ifs(jsonl_path);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "DataLoader: failed to open %s\n", jsonl_path.c_str());
        return false;
    }

    texts_.clear();
    cursor_ = 0;

    std::string line;
    size_t skipped = 0;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        try {
            auto j = nlohmann::json::parse(line);
            if (j.contains("text") && j["text"].is_string()) {
                std::string text = j["text"].get<std::string>();
                if (!text.empty()) {
                    texts_.push_back(std::move(text));
                }
            } else {
                ++skipped;
            }
        } catch (const nlohmann::json::exception&) {
            ++skipped;
        }
    }

    std::fprintf(stderr, "DataLoader: loaded %zu samples from %s",
                 texts_.size(), jsonl_path.c_str());
    if (skipped > 0) {
        std::fprintf(stderr, " (skipped %zu lines)", skipped);
    }
    std::fprintf(stderr, "\n");

    return !texts_.empty();
}

DataLoader::Batch DataLoader::next_batch(const Tokenizer& tokenizer,
                                         size_t batch_size,
                                         size_t max_seq_len) {
    Batch batch;
    if (texts_.empty()) return batch;

    for (size_t i = 0; i < batch_size; ++i) {
        // Wrap cursor on epoch boundary
        if (cursor_ >= texts_.size()) {
            cursor_ = 0;
        }

        auto tokens = tokenizer.encode(texts_[cursor_]);
        ++cursor_;

        // Truncate to max_seq_len
        if (tokens.size() > max_seq_len) {
            tokens.resize(max_seq_len);
        }

        batch.total_tokens += tokens.size();
        batch.token_sequences.push_back(std::move(tokens));
    }

    return batch;
}

void DataLoader::shuffle(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(texts_.begin(), texts_.end(), rng);
    cursor_ = 0;
}

size_t DataLoader::size() const {
    return texts_.size();
}

bool DataLoader::empty() const {
    return texts_.empty();
}

}  // namespace nos
