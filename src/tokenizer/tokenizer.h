#pragma once

/// @file tokenizer.h
/// @brief SentencePiece tokenizer wrapper.
///
/// Wraps the SentencePiece C++ API for Llama 2/Mistral .model files.
/// The interface is abstract enough to support alternative backends
/// (e.g., tiktoken BPE for Llama 3) in future phases -- a future
/// BpeTokenizer class could implement the same public API.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nos {

/// Tokenizer interface wrapping SentencePiece.
///
/// Usage:
///   nos::Tokenizer tok;
///   tok.load("path/to/llama2.model");
///   auto ids = tok.encode("Hello, world!");
///   auto text = tok.decode(ids);
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Non-copyable, movable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    /// Load a SentencePiece .model file.
    /// @param model_path  Path to the .model file
    /// @return true on success, false on failure
    bool load(const std::string& model_path);

    /// Check if a model is loaded.
    bool is_loaded() const;

    /// Encode text to token IDs.
    /// Returns empty vector if not loaded.
    std::vector<int> encode(const std::string& text) const;

    /// Decode token IDs back to text.
    /// Returns empty string if not loaded.
    std::string decode(const std::vector<int>& ids) const;

    /// Get vocabulary size. Returns 0 if not loaded.
    int vocab_size() const;

    /// Get beginning-of-sequence token ID. Returns -1 if not loaded.
    int bos_id() const;

    /// Get end-of-sequence token ID. Returns -1 if not loaded.
    int eos_id() const;

    /// Get padding token ID. Returns -1 if not loaded or not defined.
    int pad_id() const;

    /// Get unknown token ID. Returns -1 if not loaded or not defined.
    int unk_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nos
