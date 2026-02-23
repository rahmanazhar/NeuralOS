/// @file tokenizer.cpp
/// @brief SentencePiece tokenizer implementation.
///
/// Wraps sentencepiece::SentencePieceProcessor to provide encode/decode
/// for Llama 2/Mistral .model files. All methods check the loaded_ flag
/// and return graceful defaults when no model is loaded.

#include "tokenizer.h"

#include <sentencepiece_processor.h>

namespace nos {

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
    bool loaded = false;
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}

Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

bool Tokenizer::load(const std::string& model_path) {
    auto status = impl_->processor.Load(model_path);
    impl_->loaded = status.ok();
    return impl_->loaded;
}

bool Tokenizer::is_loaded() const {
    return impl_->loaded;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!impl_->loaded) {
        return {};
    }
    std::vector<int> ids;
    impl_->processor.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    if (!impl_->loaded) {
        return {};
    }
    std::string text;
    impl_->processor.Decode(ids, &text);
    return text;
}

int Tokenizer::vocab_size() const {
    if (!impl_->loaded) {
        return 0;
    }
    return impl_->processor.GetPieceSize();
}

int Tokenizer::bos_id() const {
    if (!impl_->loaded) {
        return -1;
    }
    return impl_->processor.bos_id();
}

int Tokenizer::eos_id() const {
    if (!impl_->loaded) {
        return -1;
    }
    return impl_->processor.eos_id();
}

int Tokenizer::pad_id() const {
    if (!impl_->loaded) {
        return -1;
    }
    return impl_->processor.pad_id();
}

int Tokenizer::unk_id() const {
    if (!impl_->loaded) {
        return -1;
    }
    return impl_->processor.unk_id();
}

}  // namespace nos
