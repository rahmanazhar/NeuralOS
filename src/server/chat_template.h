#pragma once

/// @file chat_template.h
/// @brief Llama-family chat template formatting for OpenAI-compatible chat API.

#include <string>
#include <utility>
#include <vector>

namespace nos {

/// Apply a chat template to format messages into a prompt string.
///
/// @param model_family  "llama3" (default) or "llama2"
/// @param messages       Vector of (role, content) pairs
/// @return Formatted prompt string ready for tokenization
std::string apply_chat_template(
    const std::string& model_family,
    const std::vector<std::pair<std::string, std::string>>& messages);

}  // namespace nos
