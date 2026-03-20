/// @file chat_template.cpp
/// @brief Llama-family chat template formatting implementation.

#include "server/chat_template.h"

#include <string>
#include <utility>
#include <vector>

namespace nos {

static std::string apply_llama3_template(
    const std::vector<std::pair<std::string, std::string>>& messages) {
    std::string result = "<|begin_of_text|>";
    for (const auto& [role, content] : messages) {
        result += "<|start_header_id|>";
        result += role;
        result += "<|end_header_id|>\n\n";
        result += content;
        result += "<|eot_id|>";
    }
    // Append the assistant header for generation
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return result;
}

static std::string apply_llama2_template(
    const std::vector<std::pair<std::string, std::string>>& messages) {
    std::string result;
    std::string system_msg;
    bool first_user = true;

    // Extract system message if present
    for (const auto& [role, content] : messages) {
        if (role == "system") {
            system_msg = content;
            break;
        }
    }

    for (const auto& [role, content] : messages) {
        if (role == "system") {
            continue;  // Already extracted above
        }

        if (role == "user") {
            result += "[INST] ";
            if (first_user && !system_msg.empty()) {
                result += "<<SYS>>\n";
                result += system_msg;
                result += "\n<</SYS>>\n\n";
            }
            result += content;
            result += " [/INST]";
            first_user = false;
        } else if (role == "assistant") {
            result += " ";
            result += content;
            result += " ";
        }
    }
    return result;
}

std::string apply_chat_template(
    const std::string& model_family,
    const std::vector<std::pair<std::string, std::string>>& messages) {
    if (messages.empty()) {
        return "";
    }

    if (model_family == "llama2") {
        return apply_llama2_template(messages);
    }
    // Default to llama3
    return apply_llama3_template(messages);
}

}  // namespace nos
