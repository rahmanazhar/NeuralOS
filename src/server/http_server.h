#pragma once

/// @file http_server.h
/// @brief HTTP server wrapping cpp-httplib for OpenAI-compatible inference API.

#include "api/libneuralos.h"

#include <memory>
#include <string>

namespace nos {

class HttpServer {
public:
    struct Config {
        std::string host = "127.0.0.1";
        int port = 8080;
        std::string model_path;
        nos_config_t inference_config{};
    };

    HttpServer();
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    /// Load model and start listening. Blocks in server thread.
    /// @return true if server started successfully
    bool start(const Config& config);

    /// Stop the server gracefully.
    void stop();

    /// Check if server is currently running.
    bool is_running() const;

    // ── Static helpers for JSON response formatting (public for testing) ──

    /// Build an OpenAI-compatible completion response JSON string.
    static std::string build_completion_response(
        const std::string& id,
        const std::string& model,
        long created,
        const std::string& text,
        const std::string& finish_reason,
        int prompt_tokens,
        int completion_tokens);

    /// Build an OpenAI-compatible chat completion response JSON string.
    static std::string build_chat_completion_response(
        const std::string& id,
        const std::string& model,
        long created,
        const std::string& content,
        const std::string& finish_reason,
        int prompt_tokens,
        int completion_tokens);

    /// Build a streaming completion chunk JSON string.
    static std::string build_completion_chunk(
        const std::string& id,
        const std::string& model,
        long created,
        const std::string& text,
        const char* finish_reason);

    /// Build a streaming chat completion chunk JSON string.
    static std::string build_chat_completion_chunk(
        const std::string& id,
        const std::string& model,
        long created,
        const std::string& content,
        const char* finish_reason);

    /// Build an error response JSON string.
    static std::string build_error_response(
        const std::string& message,
        const std::string& type,
        int code);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nos
