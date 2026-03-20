/// @file test_http_server.cpp
/// @brief Catch2 tests for HTTP server JSON response formatting.
///
/// These are unit-level tests that validate JSON structure without
/// starting an actual server. Full integration tests would require
/// running the server in a thread.

#include <catch2/catch_test_macros.hpp>

#include <string>

#include <nlohmann/json.hpp>

#include "server/http_server.h"

// ── Completion Response Tests ───────────────────────────────────────────────

TEST_CASE("Completion response has required OpenAI fields",
          "[http_server][response]") {
    std::string json_str = nos::HttpServer::build_completion_response(
        "nos-12345", "test-model", 1700000000,
        "Hello world", "stop", 5, 2);

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j.contains("id"));
    REQUIRE(j.contains("object"));
    REQUIRE(j.contains("created"));
    REQUIRE(j.contains("model"));
    REQUIRE(j.contains("choices"));
    REQUIRE(j.contains("usage"));

    REQUIRE(j["id"] == "nos-12345");
    REQUIRE(j["object"] == "text_completion");
    REQUIRE(j["created"] == 1700000000);
    REQUIRE(j["model"] == "test-model");

    REQUIRE(j["choices"].is_array());
    REQUIRE(j["choices"].size() == 1);
    REQUIRE(j["choices"][0]["index"] == 0);
    REQUIRE(j["choices"][0]["text"] == "Hello world");
    REQUIRE(j["choices"][0]["finish_reason"] == "stop");

    REQUIRE(j["usage"]["prompt_tokens"] == 5);
    REQUIRE(j["usage"]["completion_tokens"] == 2);
    REQUIRE(j["usage"]["total_tokens"] == 7);
}

// ── Chat Completion Response Tests ──────────────────────────────────────────

TEST_CASE("Chat completion response has correct structure",
          "[http_server][response]") {
    std::string json_str = nos::HttpServer::build_chat_completion_response(
        "nos-67890", "llama-7b", 1700000001,
        "I am an AI assistant.", "stop", 10, 5);

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j["object"] == "chat.completion");
    REQUIRE(j["choices"][0]["message"]["role"] == "assistant");
    REQUIRE(j["choices"][0]["message"]["content"] == "I am an AI assistant.");
    REQUIRE(j["choices"][0]["finish_reason"] == "stop");
}

// ── Streaming Chunk Tests ───────────────────────────────────────────────────

TEST_CASE("Streaming completion chunk format", "[http_server][streaming]") {
    std::string json_str = nos::HttpServer::build_completion_chunk(
        "nos-stream-1", "test-model", 1700000002, "Hello", nullptr);

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j["id"] == "nos-stream-1");
    REQUIRE(j["object"] == "text_completion.chunk");
    REQUIRE(j["choices"][0]["text"] == "Hello");
    REQUIRE(j["choices"][0]["finish_reason"].is_null());
}

TEST_CASE("Streaming completion final chunk has stop reason",
          "[http_server][streaming]") {
    std::string json_str = nos::HttpServer::build_completion_chunk(
        "nos-stream-2", "test-model", 1700000003, "", "stop");

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j["choices"][0]["finish_reason"] == "stop");
}

TEST_CASE("Streaming chat completion chunk format", "[http_server][streaming]") {
    std::string json_str = nos::HttpServer::build_chat_completion_chunk(
        "nos-chat-1", "llama-7b", 1700000004, "token", nullptr);

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j["object"] == "chat.completion.chunk");
    REQUIRE(j["choices"][0]["delta"]["content"] == "token");
    REQUIRE(j["choices"][0]["finish_reason"].is_null());
}

// ── Error Response Tests ────────────────────────────────────────────────────

TEST_CASE("Error response format", "[http_server][error]") {
    std::string json_str = nos::HttpServer::build_error_response(
        "Invalid JSON body", "invalid_request_error", 400);

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j.contains("error"));
    REQUIRE(j["error"]["message"] == "Invalid JSON body");
    REQUIRE(j["error"]["type"] == "invalid_request_error");
    REQUIRE(j["error"]["code"] == 400);
}

TEST_CASE("Server error response format", "[http_server][error]") {
    std::string json_str = nos::HttpServer::build_error_response(
        "Model inference failed", "server_error", 500);

    auto j = nlohmann::json::parse(json_str);

    REQUIRE(j["error"]["message"] == "Model inference failed");
    REQUIRE(j["error"]["type"] == "server_error");
    REQUIRE(j["error"]["code"] == 500);
}
