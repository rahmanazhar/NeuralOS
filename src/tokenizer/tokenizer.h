#pragma once

/// @file tokenizer.h
/// @brief SentencePiece tokenizer wrapper.
///
/// Wraps the SentencePiece C++ API for Llama 2/Mistral .model files.
/// The interface is abstract enough to support alternative backends
/// (e.g., tiktoken BPE for Llama 3) in future phases.

#include <cstdint>
#include <string>
#include <vector>

namespace nos {

/// Tokenizer interface wrapping SentencePiece.
class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // Non-copyable, movable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = default;
    Tokenizer& operator=(Tokenizer&&) = default;

    /// Load a SentencePiece .model file.
    ///
    /// @param model_path  Path to the .model file
    /// @return true on success, false on failure
    bool load(const std::string& model_path);

    /// Encode text to token IDs.
    ///
    /// @param text  Input text
    /// @return Vector of token IDs
    std::vector<int> encode(const std::string& text) const;

    /// Decode token IDs back to text.
    ///
    /// @param ids  Vector of token IDs
    /// @return Decoded text
    std::string decode(const std::vector<int>& ids) const;

    /// Get vocabulary size.
    int vocab_size() const;

    /// Get beginning-of-sequence token ID.
    int bos_id() const;

    /// Get end-of-sequence token ID.
    int eos_id() const;
};

}  // namespace nos
