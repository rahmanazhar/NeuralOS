/// @file tokenizer_stub.cpp
/// @brief Placeholder source for nos_tokenizer library. Full implementation in Plan 04.

#include "tokenizer.h"

namespace nos {

bool Tokenizer::load(const std::string& /*model_path*/) { return false; }

std::vector<int> Tokenizer::encode(const std::string& /*text*/) const { return {}; }

std::string Tokenizer::decode(const std::vector<int>& /*ids*/) const { return {}; }

int Tokenizer::vocab_size() const { return 0; }

int Tokenizer::bos_id() const { return -1; }

int Tokenizer::eos_id() const { return -1; }

}  // namespace nos
