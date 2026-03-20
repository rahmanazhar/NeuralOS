/// @file http_server.cpp
/// @brief HTTP server implementation with OpenAI-compatible endpoints, SSE streaming,
///        multi-sequence batching via RequestScheduler, and shared metrics via MetricsWriter.

#include "server/http_server.h"
#include "server/chat_template.h"
#include "server/request_scheduler.h"
#include "server/shared_metrics.h"
#include "api/libneuralos.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>  // getpid

namespace nos {

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::string generate_request_id() {
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return "nos-" + std::to_string(epoch);
}

static long unix_timestamp() {
    auto now = std::chrono::system_clock::now();
    return static_cast<long>(
        std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count());
}

static double epoch_seconds() {
    auto now = std::chrono::system_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count()) / 1000.0;
}

// ── JSON Response Builders ──────────────────────────────────────────────────

std::string HttpServer::build_completion_response(
    const std::string& id,
    const std::string& model,
    long created,
    const std::string& text,
    const std::string& finish_reason,
    int prompt_tokens,
    int completion_tokens) {
    nlohmann::json j;
    j["id"] = id;
    j["object"] = "text_completion";
    j["created"] = created;
    j["model"] = model;
    j["choices"] = nlohmann::json::array({
        {
            {"index", 0},
            {"text", text},
            {"finish_reason", finish_reason}
        }
    });
    j["usage"] = {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + completion_tokens}
    };
    return j.dump();
}

std::string HttpServer::build_chat_completion_response(
    const std::string& id,
    const std::string& model,
    long created,
    const std::string& content,
    const std::string& finish_reason,
    int prompt_tokens,
    int completion_tokens) {
    nlohmann::json j;
    j["id"] = id;
    j["object"] = "chat.completion";
    j["created"] = created;
    j["model"] = model;
    j["choices"] = nlohmann::json::array({
        {
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", content}}},
            {"finish_reason", finish_reason}
        }
    });
    j["usage"] = {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + completion_tokens}
    };
    return j.dump();
}

std::string HttpServer::build_completion_chunk(
    const std::string& id,
    const std::string& model,
    long created,
    const std::string& text,
    const char* finish_reason) {
    nlohmann::json j;
    j["id"] = id;
    j["object"] = "text_completion.chunk";
    j["created"] = created;
    j["model"] = model;
    j["choices"] = nlohmann::json::array({
        {
            {"index", 0},
            {"text", text},
            {"finish_reason", finish_reason ? nlohmann::json(finish_reason)
                                            : nlohmann::json(nullptr)}
        }
    });
    return j.dump();
}

std::string HttpServer::build_chat_completion_chunk(
    const std::string& id,
    const std::string& model,
    long created,
    const std::string& content,
    const char* finish_reason) {
    nlohmann::json j;
    j["id"] = id;
    j["object"] = "chat.completion.chunk";
    j["created"] = created;
    j["model"] = model;
    j["choices"] = nlohmann::json::array({
        {
            {"index", 0},
            {"delta", {{"content", content}}},
            {"finish_reason", finish_reason ? nlohmann::json(finish_reason)
                                            : nlohmann::json(nullptr)}
        }
    });
    return j.dump();
}

std::string HttpServer::build_error_response(
    const std::string& message,
    const std::string& type,
    int code) {
    nlohmann::json j;
    j["error"] = {
        {"message", message},
        {"type", type},
        {"code", code}
    };
    return j.dump();
}

// ── CORS Helper ─────────────────────────────────────────────────────────────

static void set_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

// ── Server Implementation ───────────────────────────────────────────────────

struct HttpServer::Impl {
    httplib::Server svr;
    std::unique_ptr<RequestScheduler> scheduler;
    std::unique_ptr<MetricsWriter> metrics_writer;
    std::string model_path;
    std::string shm_name;
    std::atomic<bool> running{false};
    std::thread server_thread;

    // Throttle metrics writes to every 100ms
    std::chrono::steady_clock::time_point last_metrics_write{};
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start(const Config& config) {
    if (impl_->running.load()) return false;

    impl_->model_path = config.model_path;

    // Create request scheduler with multiple slots
    impl_->scheduler = std::make_unique<RequestScheduler>(
        config.max_slots, config.model_path, config.inference_config);

    if (!impl_->scheduler->is_ready()) {
        std::fprintf(stderr, "HttpServer: Failed to create any inference slots: %s\n",
                     nos_last_error());
        return false;
    }

    // Create shared memory metrics writer
    impl_->shm_name = "/neuralos_metrics_" + std::to_string(getpid());
    impl_->metrics_writer = std::make_unique<MetricsWriter>(impl_->shm_name);
    if (!impl_->metrics_writer->is_open()) {
        std::fprintf(stderr, "HttpServer: Warning: Could not create shared metrics segment\n");
        // Non-fatal -- server can run without metrics
    }

    // ── OPTIONS preflight ───────────────────────────────────────────────
    impl_->svr.Options(".*", [](const httplib::Request& /*req*/,
                                httplib::Response& res) {
        set_cors_headers(res);
        res.status = 204;
    });

    // ── GET /health ─────────────────────────────────────────────────────
    impl_->svr.Get("/health", [this](const httplib::Request& /*req*/,
                                     httplib::Response& res) {
        set_cors_headers(res);
        nlohmann::json j;
        if (impl_->scheduler && impl_->scheduler->is_ready()) {
            j["status"] = "ready";
            j["model"] = impl_->model_path;
            j["active_slots"] = impl_->scheduler->active_count();
            j["max_slots"] = impl_->scheduler->slot_count();
            j["shm_name"] = impl_->shm_name;
            res.status = 200;
        } else {
            j["status"] = "error";
            res.status = 503;
        }
        res.set_content(j.dump(), "application/json");
    });

    // ── POST /v1/completions ────────────────────────────────────────────
    impl_->svr.Post("/v1/completions",
        [this](const httplib::Request& req, httplib::Response& res) {
        set_cors_headers(res);

        // Parse JSON body
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(
                build_error_response("Invalid JSON body", "invalid_request_error", 400),
                "application/json");
            return;
        }

        if (!body.contains("prompt") || !body["prompt"].is_string()) {
            res.status = 400;
            res.set_content(
                build_error_response("Missing 'prompt' field", "invalid_request_error", 400),
                "application/json");
            return;
        }

        std::string prompt = body["prompt"].get<std::string>();
        int max_tokens = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        std::string req_id = generate_request_id();
        long ts = unix_timestamp();

        // Acquire a slot from the scheduler
        auto guard = impl_->scheduler->acquire_slot_guard();
        if (!guard) {
            res.status = 503;
            res.set_content(
                build_error_response("Server busy -- all inference slots in use",
                                     "server_error", 503),
                "application/json");
            return;
        }

        nos_ctx_t* ctx = guard->ctx;

        if (!stream) {
            // Non-streaming: use nos_generate
            nos_reset(ctx);

            std::vector<char> buf(static_cast<size_t>(max_tokens) * 16 + 1);
            int rc = nos_generate(ctx, prompt.c_str(),
                                  buf.data(), buf.size());
            if (rc != NOS_OK && rc != NOS_ERR_BUFFER) {
                res.status = 500;
                res.set_content(
                    build_error_response(nos_last_error(), "server_error", 500),
                    "application/json");
                return;
            }

            std::string text(buf.data());

            // Count prompt tokens
            int prompt_ids[4096];
            size_t num_prompt = 0;
            nos_tokenize(ctx, prompt.c_str(), prompt_ids, 4096, &num_prompt);
            int prompt_tok = static_cast<int>(num_prompt);

            // Count completion tokens
            int comp_ids[4096];
            size_t num_comp = 0;
            nos_tokenize(ctx, text.c_str(), comp_ids, 4096, &num_comp);
            int comp_tok = static_cast<int>(num_comp);

            // Update shared metrics (throttled)
            update_shared_metrics();

            res.status = 200;
            res.set_content(
                build_completion_response(req_id, impl_->model_path, ts,
                                          text, "stop", prompt_tok, comp_tok),
                "application/json");
        } else {
            // Streaming: SSE via chunked content provider
            nos_reset(ctx);

            // Tokenize prompt first
            int prompt_ids[4096];
            size_t num_prompt = 0;
            nos_tokenize(ctx, prompt.c_str(),
                         prompt_ids, 4096, &num_prompt);

            // Step through prompt tokens
            int out_token = 0;
            for (size_t i = 0; i < num_prompt; ++i) {
                nos_step_token(ctx, prompt_ids[i], &out_token);
            }

            // Capture state for the chunked provider
            struct StreamState {
                nos_ctx_t* ctx;
                std::string req_id;
                std::string model;
                long ts;
                int max_tokens;
                int generated;
                bool done;
            };
            auto state = std::make_shared<StreamState>();
            state->ctx = ctx;
            state->req_id = req_id;
            state->model = impl_->model_path;
            state->ts = ts;
            state->max_tokens = max_tokens;
            state->generated = 0;
            state->done = false;

            // The last out_token from prompt processing is the first gen token
            int first_token = out_token;

            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [state, first_token](size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                    if (state->done) return false;

                    // For the first call, decode the first generated token
                    int token_to_decode = first_token;
                    if (state->generated > 0) {
                        // Step to get next token
                        int next = 0;
                        int rc = nos_step_token(state->ctx, token_to_decode, &next);
                        if (rc != NOS_OK) {
                            state->done = true;
                            std::string final_chunk = "data: " +
                                HttpServer::build_completion_chunk(
                                    state->req_id, state->model, state->ts, "", "stop") + "\n\n";
                            if (!sink.write(final_chunk.data(), final_chunk.size())) return false;
                            if (!sink.write("data: [DONE]\n\n", 14)) return false;
                            return false;
                        }
                        token_to_decode = next;
                    }

                    // Decode token to text
                    char token_buf[256];
                    int ids[] = {token_to_decode};
                    nos_detokenize(state->ctx, ids, 1, token_buf, sizeof(token_buf));
                    std::string token_text(token_buf);

                    // Send chunk
                    std::string chunk = "data: " +
                        HttpServer::build_completion_chunk(
                            state->req_id, state->model, state->ts,
                            token_text, nullptr) + "\n\n";
                    if (!sink.write(chunk.data(), chunk.size())) return false;

                    state->generated++;
                    first_token = token_to_decode;  // For next iteration's nos_step_token

                    if (state->generated >= state->max_tokens) {
                        std::string final_chunk = "data: " +
                            HttpServer::build_completion_chunk(
                                state->req_id, state->model, state->ts, "", "stop") + "\n\n";
                        if (!sink.write(final_chunk.data(), final_chunk.size())) return false;
                        if (!sink.write("data: [DONE]\n\n", 14)) return false;
                        state->done = true;
                        return false;
                    }
                    return true;
                },
                [](bool /*success*/) {
                    // Cleanup callback -- nothing needed
                }
            );
        }
    });

    // ── POST /v1/chat/completions ───────────────────────────────────────
    impl_->svr.Post("/v1/chat/completions",
        [this](const httplib::Request& req, httplib::Response& res) {
        set_cors_headers(res);

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(
                build_error_response("Invalid JSON body", "invalid_request_error", 400),
                "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(
                build_error_response("Missing 'messages' array", "invalid_request_error", 400),
                "application/json");
            return;
        }

        // Parse messages
        std::vector<std::pair<std::string, std::string>> messages;
        for (const auto& m : body["messages"]) {
            if (m.contains("role") && m.contains("content")) {
                messages.emplace_back(
                    m["role"].get<std::string>(),
                    m["content"].get<std::string>());
            }
        }

        // Apply chat template
        std::string model_family = body.value("model_family", "llama3");
        std::string prompt = apply_chat_template(model_family, messages);

        int max_tokens = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        std::string req_id = generate_request_id();
        long ts = unix_timestamp();

        // Acquire a slot from the scheduler
        auto guard = impl_->scheduler->acquire_slot_guard();
        if (!guard) {
            res.status = 503;
            res.set_content(
                build_error_response("Server busy -- all inference slots in use",
                                     "server_error", 503),
                "application/json");
            return;
        }

        nos_ctx_t* ctx = guard->ctx;

        if (!stream) {
            nos_reset(ctx);

            std::vector<char> buf(static_cast<size_t>(max_tokens) * 16 + 1);
            int rc = nos_generate(ctx, prompt.c_str(),
                                  buf.data(), buf.size());
            if (rc != NOS_OK && rc != NOS_ERR_BUFFER) {
                res.status = 500;
                res.set_content(
                    build_error_response(nos_last_error(), "server_error", 500),
                    "application/json");
                return;
            }

            std::string content(buf.data());

            // Count tokens
            int prompt_ids[4096];
            size_t num_prompt = 0;
            nos_tokenize(ctx, prompt.c_str(), prompt_ids, 4096, &num_prompt);
            int prompt_tok = static_cast<int>(num_prompt);

            int comp_ids[4096];
            size_t num_comp = 0;
            nos_tokenize(ctx, content.c_str(), comp_ids, 4096, &num_comp);
            int comp_tok = static_cast<int>(num_comp);

            // Update shared metrics (throttled)
            update_shared_metrics();

            res.status = 200;
            res.set_content(
                build_chat_completion_response(req_id, impl_->model_path, ts,
                                               content, "stop", prompt_tok, comp_tok),
                "application/json");
        } else {
            nos_reset(ctx);

            int prompt_ids[4096];
            size_t num_prompt = 0;
            nos_tokenize(ctx, prompt.c_str(),
                         prompt_ids, 4096, &num_prompt);

            int out_token = 0;
            for (size_t i = 0; i < num_prompt; ++i) {
                nos_step_token(ctx, prompt_ids[i], &out_token);
            }

            struct StreamState {
                nos_ctx_t* ctx;
                std::string req_id;
                std::string model;
                long ts;
                int max_tokens;
                int generated;
                bool done;
            };
            auto state = std::make_shared<StreamState>();
            state->ctx = ctx;
            state->req_id = req_id;
            state->model = impl_->model_path;
            state->ts = ts;
            state->max_tokens = max_tokens;
            state->generated = 0;
            state->done = false;

            int first_token = out_token;

            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [state, first_token](size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
                    if (state->done) return false;

                    int token_to_decode = first_token;
                    if (state->generated > 0) {
                        int next = 0;
                        int rc = nos_step_token(state->ctx, token_to_decode, &next);
                        if (rc != NOS_OK) {
                            state->done = true;
                            std::string final_chunk = "data: " +
                                HttpServer::build_chat_completion_chunk(
                                    state->req_id, state->model, state->ts, "", "stop") + "\n\n";
                            if (!sink.write(final_chunk.data(), final_chunk.size())) return false;
                            if (!sink.write("data: [DONE]\n\n", 14)) return false;
                            return false;
                        }
                        token_to_decode = next;
                    }

                    char token_buf[256];
                    int ids[] = {token_to_decode};
                    nos_detokenize(state->ctx, ids, 1, token_buf, sizeof(token_buf));
                    std::string token_text(token_buf);

                    std::string chunk = "data: " +
                        HttpServer::build_chat_completion_chunk(
                            state->req_id, state->model, state->ts,
                            token_text, nullptr) + "\n\n";
                    if (!sink.write(chunk.data(), chunk.size())) return false;

                    state->generated++;
                    first_token = token_to_decode;

                    if (state->generated >= state->max_tokens) {
                        std::string final_chunk = "data: " +
                            HttpServer::build_chat_completion_chunk(
                                state->req_id, state->model, state->ts, "", "stop") + "\n\n";
                        if (!sink.write(final_chunk.data(), final_chunk.size())) return false;
                        if (!sink.write("data: [DONE]\n\n", 14)) return false;
                        state->done = true;
                        return false;
                    }
                    return true;
                },
                [](bool /*success*/) {}
            );
        }
    });

    // Start server in a thread
    impl_->running.store(true);
    impl_->server_thread = std::thread([this, config]() {
        impl_->svr.listen(config.host, config.port);
        impl_->running.store(false);
    });

    // Give server a moment to bind
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return impl_->svr.is_running();
}

void HttpServer::stop() {
    if (impl_->svr.is_running()) {
        impl_->svr.stop();
    }
    if (impl_->server_thread.joinable()) {
        impl_->server_thread.join();
    }
    // Scheduler destructor handles nos_destroy for all slots
    impl_->scheduler.reset();
    impl_->metrics_writer.reset();
    impl_->running.store(false);
}

bool HttpServer::is_running() const {
    return impl_->running.load();
}

void HttpServer::update_shared_metrics() {
    if (!impl_->metrics_writer || !impl_->metrics_writer->is_open()) return;

    // Throttle to every 100ms
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - impl_->last_metrics_write);
    if (elapsed.count() < 100) return;
    impl_->last_metrics_write = now;

    SharedMetrics m{};
    m.last_update_epoch = epoch_seconds();
    m.active_slots = static_cast<uint32_t>(impl_->scheduler->active_count());
    m.max_slots = static_cast<uint32_t>(impl_->scheduler->slot_count());

    impl_->metrics_writer->update(m);
}

}  // namespace nos
