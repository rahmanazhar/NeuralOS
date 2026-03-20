/// @file test_chat_template.cpp
/// @brief Catch2 tests for Llama-family chat template formatting.

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <utility>
#include <vector>

#include "server/chat_template.h"

using Messages = std::vector<std::pair<std::string, std::string>>;

// ── Llama 3 Template Tests ──────────────────────────────────────────────────

TEST_CASE("Llama 3 template formats system + user + assistant correctly",
          "[chat_template][llama3]") {
    Messages msgs = {
        {"system", "You are a helpful assistant."},
        {"user", "Hello!"},
        {"assistant", "Hi there!"},
        {"user", "How are you?"}
    };

    std::string result = nos::apply_chat_template("llama3", msgs);

    // Should start with begin_of_text
    REQUIRE(result.find("<|begin_of_text|>") == 0);

    // Should contain all messages with proper header format
    REQUIRE(result.find("<|start_header_id|>system<|end_header_id|>\n\n"
                        "You are a helpful assistant.<|eot_id|>") != std::string::npos);
    REQUIRE(result.find("<|start_header_id|>user<|end_header_id|>\n\n"
                        "Hello!<|eot_id|>") != std::string::npos);
    REQUIRE(result.find("<|start_header_id|>assistant<|end_header_id|>\n\n"
                        "Hi there!<|eot_id|>") != std::string::npos);
    REQUIRE(result.find("<|start_header_id|>user<|end_header_id|>\n\n"
                        "How are you?<|eot_id|>") != std::string::npos);

    // Should end with assistant header for generation
    std::string expected_suffix = "<|start_header_id|>assistant<|end_header_id|>\n\n";
    REQUIRE(result.substr(result.size() - expected_suffix.size()) == expected_suffix);
}

TEST_CASE("Llama 3 template with user-only message (no system)",
          "[chat_template][llama3]") {
    Messages msgs = {
        {"user", "What is 2+2?"}
    };

    std::string result = nos::apply_chat_template("llama3", msgs);

    REQUIRE(result.find("<|begin_of_text|>") == 0);
    REQUIRE(result.find("<|start_header_id|>user<|end_header_id|>\n\n"
                        "What is 2+2?<|eot_id|>") != std::string::npos);

    // Should not contain system header
    REQUIRE(result.find("system") == std::string::npos);

    // Should end with assistant header
    std::string expected_suffix = "<|start_header_id|>assistant<|end_header_id|>\n\n";
    REQUIRE(result.substr(result.size() - expected_suffix.size()) == expected_suffix);
}

// ── Llama 2 Template Tests ──────────────────────────────────────────────────

TEST_CASE("Llama 2 template basic formatting", "[chat_template][llama2]") {
    Messages msgs = {
        {"system", "You are a helpful AI."},
        {"user", "Hello!"}
    };

    std::string result = nos::apply_chat_template("llama2", msgs);

    REQUIRE(result.find("[INST]") != std::string::npos);
    REQUIRE(result.find("<<SYS>>") != std::string::npos);
    REQUIRE(result.find("You are a helpful AI.") != std::string::npos);
    REQUIRE(result.find("<</SYS>>") != std::string::npos);
    REQUIRE(result.find("Hello!") != std::string::npos);
    REQUIRE(result.find("[/INST]") != std::string::npos);
}

TEST_CASE("Llama 2 template without system message", "[chat_template][llama2]") {
    Messages msgs = {
        {"user", "Hi there"}
    };

    std::string result = nos::apply_chat_template("llama2", msgs);

    REQUIRE(result.find("[INST]") != std::string::npos);
    REQUIRE(result.find("Hi there") != std::string::npos);
    REQUIRE(result.find("[/INST]") != std::string::npos);
    // Should NOT contain SYS markers
    REQUIRE(result.find("<<SYS>>") == std::string::npos);
}

// ── Edge Cases ──────────────────────────────────────────────────────────────

TEST_CASE("Empty messages handled gracefully", "[chat_template]") {
    Messages msgs = {};
    std::string result = nos::apply_chat_template("llama3", msgs);
    REQUIRE(result.empty());
}

TEST_CASE("Default model family is llama3", "[chat_template]") {
    Messages msgs = {{"user", "test"}};

    std::string llama3 = nos::apply_chat_template("llama3", msgs);
    std::string unknown = nos::apply_chat_template("unknown_model", msgs);

    // Unknown model family should fall back to llama3 format
    REQUIRE(llama3 == unknown);
}
