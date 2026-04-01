/// @file libneuralos.cpp
/// @brief C API implementation wrapping InferenceEngine, Tokenizer, and Vmm.

#include "api/libneuralos.h"

#include "engine/inference_engine.h"
#include "engine/sampling.h"
#include "tokenizer/tokenizer.h"
#include "vmm/memory_budget.h"
#include "vmm/vmm.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// ── Thread-local error storage ──────────────────────────────────────────────

static thread_local std::string tl_last_error;

static void set_error(const std::string& msg) {
    tl_last_error = msg;
}

static void clear_error() {
    tl_last_error.clear();
}

// ── Helper: find tokenizer file in model directory ──────────────────────────

static std::string find_tokenizer(const std::string& dir) {
    std::vector<std::string> candidates = {
        dir + "/tokenizer.model",
        dir + "/tokenizer.json",
    };
    for (const auto& c : candidates) {
        std::ifstream f(c);
        if (f.good()) return c;
    }
    return "";
}

// ── Internal context struct ─────────────────────────────────────────────────

struct nos_ctx {
    nos::InferenceEngine engine;
    nos::Tokenizer tokenizer;
    std::unique_ptr<nos::Vmm> vmm;
    nos::SamplingParams sampling_params;
    std::mt19937 rng;
    std::string metrics_json_cache;
    std::vector<int> context;
    int current_pos = 0;
    std::string model_path;
};

// ── Lifecycle ───────────────────────────────────────────────────────────────

nos_ctx_t* nos_create(const char* model_path, nos_config_t config) {
    clear_error();

    if (model_path == nullptr) {
        set_error("model_path is NULL");
        return nullptr;
    }

    if (config.struct_size != sizeof(nos_config_t)) {
        set_error("nos_config_t struct_size mismatch (ABI version error): expected "
                  + std::to_string(sizeof(nos_config_t))
                  + ", got " + std::to_string(config.struct_size));
        return nullptr;
    }

    try {
        auto ctx = std::make_unique<nos_ctx>();
        ctx->model_path = model_path;

        // Set up memory budget
        size_t budget = config.memory_budget;
        if (budget == 0) {
            budget = nos::parse_memory_string("8G");
        }

        // Create VMM
        nos::VmmFullConfig vmm_cfg;
        vmm_cfg.nxp_path = std::string(model_path) + "/model.nxp";
        vmm_cfg.user_budget_bytes = budget;
        ctx->vmm = nos::Vmm::create(vmm_cfg);
        if (!ctx->vmm) {
            set_error("Failed to create VMM for model at: " + std::string(model_path));
            return nullptr;
        }

        // Configure sticky routing
        ctx->engine.set_sticky_config(config.sticky_lambda, config.sticky_window);

        // Configure prefetch
        ctx->engine.set_prefetch_config(config.prefetch_enabled != 0,
                                        config.prefetch_max_k);

        // Load engine
        if (!ctx->engine.load(model_path, ctx->vmm.get(), config.num_threads)) {
            set_error("Failed to load model from: " + std::string(model_path));
            return nullptr;
        }

        // Load tokenizer
        std::string tok_path = find_tokenizer(model_path);
        if (!tok_path.empty()) {
            ctx->tokenizer.load(tok_path);
        }

        // Set sampling params
        ctx->sampling_params.temperature = config.temperature;
        ctx->sampling_params.top_k = config.top_k;
        ctx->sampling_params.top_p = config.top_p;
        ctx->sampling_params.repetition_penalty = config.repetition_penalty;
        ctx->sampling_params.min_p = config.min_p;

        // Initialize RNG
        std::random_device rd;
        ctx->rng.seed(rd());

        return ctx.release();
    } catch (const std::exception& e) {
        set_error(std::string("Exception in nos_create: ") + e.what());
        return nullptr;
    } catch (...) {
        set_error("Unknown exception in nos_create");
        return nullptr;
    }
}

void nos_destroy(nos_ctx_t* ctx) {
    delete ctx;
}

int nos_reset(nos_ctx_t* ctx) {
    clear_error();
    if (ctx == nullptr) {
        set_error("nos_reset: ctx is NULL");
        return NOS_ERR_INVALID;
    }
    try {
        ctx->engine.reset_kv_cache();
        ctx->context.clear();
        ctx->current_pos = 0;
        return NOS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("nos_reset: ") + e.what());
        return NOS_ERR_INTERNAL;
    }
}

// ── Inference ───────────────────────────────────────────────────────────────

int nos_step_token(nos_ctx_t* ctx, int token_id, int* out_token) {
    clear_error();
    if (ctx == nullptr || out_token == nullptr) {
        set_error("nos_step_token: ctx or out_token is NULL");
        return NOS_ERR_INVALID;
    }
    try {
        const float* logits = ctx->engine.forward_step(token_id, ctx->current_pos);
        if (logits == nullptr) {
            set_error("nos_step_token: forward_step returned null logits");
            return NOS_ERR_MODEL;
        }
        ctx->current_pos++;
        ctx->context.push_back(token_id);

        // Sample next token
        std::vector<float> logits_copy(logits,
            logits + ctx->engine.vocab_size());
        int next = nos::sample(logits_copy.data(), ctx->engine.vocab_size(),
                               ctx->sampling_params, ctx->context, ctx->rng);
        *out_token = next;
        return NOS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("nos_step_token: ") + e.what());
        return NOS_ERR_INTERNAL;
    }
}

int nos_step_text(nos_ctx_t* ctx, const char* text, char* out_buf, size_t buf_len) {
    clear_error();
    if (ctx == nullptr || text == nullptr || out_buf == nullptr) {
        set_error("nos_step_text: ctx, text, or out_buf is NULL");
        return NOS_ERR_INVALID;
    }
    if (buf_len == 0) {
        set_error("nos_step_text: buf_len is 0");
        return NOS_ERR_BUFFER;
    }
    try {
        // Tokenize input text
        std::vector<int> ids;
        if (ctx->tokenizer.is_loaded()) {
            ids = ctx->tokenizer.encode(text);
        } else {
            // Byte-level fallback
            const char* p = text;
            while (*p != '\0') {
                ids.push_back(static_cast<int>(static_cast<unsigned char>(*p)));
                ++p;
            }
        }

        if (ids.empty()) {
            out_buf[0] = '\0';
            return NOS_OK;
        }

        // Forward each token
        const float* logits = nullptr;
        for (size_t i = 0; i < ids.size(); ++i) {
            logits = ctx->engine.forward_step(ids[static_cast<size_t>(i)],
                                              ctx->current_pos);
            ctx->current_pos++;
            ctx->context.push_back(ids[static_cast<size_t>(i)]);
        }

        if (logits == nullptr) {
            set_error("nos_step_text: forward_step returned null logits");
            return NOS_ERR_MODEL;
        }

        // Sample
        std::vector<float> logits_copy(logits,
            logits + ctx->engine.vocab_size());
        int next = nos::sample(logits_copy.data(), ctx->engine.vocab_size(),
                               ctx->sampling_params, ctx->context, ctx->rng);

        // Decode output token
        std::string decoded;
        if (ctx->tokenizer.is_loaded()) {
            decoded = ctx->tokenizer.decode({next});
        } else {
            if (next >= 0 && next < 128) {
                decoded = std::string(1, static_cast<char>(next));
            }
        }

        if (decoded.size() >= buf_len) {
            set_error("nos_step_text: output buffer too small");
            return NOS_ERR_BUFFER;
        }

        std::memcpy(out_buf, decoded.data(), decoded.size());
        out_buf[decoded.size()] = '\0';
        return NOS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("nos_step_text: ") + e.what());
        return NOS_ERR_INTERNAL;
    }
}

int nos_generate(nos_ctx_t* ctx, const char* prompt, char* out_buf, size_t buf_len) {
    clear_error();
    if (ctx == nullptr || prompt == nullptr || out_buf == nullptr) {
        set_error("nos_generate: ctx, prompt, or out_buf is NULL");
        return NOS_ERR_INVALID;
    }
    if (buf_len == 0) {
        set_error("nos_generate: buf_len is 0");
        return NOS_ERR_BUFFER;
    }
    try {
        // Tokenize prompt
        std::vector<int> prompt_ids;
        if (ctx->tokenizer.is_loaded()) {
            prompt_ids = ctx->tokenizer.encode(prompt);
            if (ctx->tokenizer.bos_id() >= 0) {
                prompt_ids.insert(prompt_ids.begin(), ctx->tokenizer.bos_id());
            }
        } else {
            const char* p = prompt;
            while (*p != '\0') {
                prompt_ids.push_back(
                    static_cast<int>(static_cast<unsigned char>(*p)));
                ++p;
            }
        }

        // Prefill: process prompt tokens
        const float* logits = nullptr;
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            logits = ctx->engine.forward_step(
                prompt_ids[i], ctx->current_pos);
            ctx->current_pos++;
            ctx->context.push_back(prompt_ids[i]);
        }

        if (logits == nullptr) {
            out_buf[0] = '\0';
            return NOS_OK;
        }

        // Generate tokens until EOS or buffer full
        int eos_id = ctx->tokenizer.is_loaded() ? ctx->tokenizer.eos_id() : -1;
        std::string generated;
        int max_seq = ctx->engine.config().max_seq_len > 0
            ? static_cast<int>(ctx->engine.config().max_seq_len)
            : 2048;
        int max_gen = max_seq - static_cast<int>(prompt_ids.size());
        if (max_gen <= 0) max_gen = 256;

        for (int t = 0; t < max_gen; ++t) {
            std::vector<float> logits_copy(logits,
                logits + ctx->engine.vocab_size());
            int next = nos::sample(logits_copy.data(),
                                   ctx->engine.vocab_size(),
                                   ctx->sampling_params, ctx->context,
                                   ctx->rng);

            if (next == eos_id) break;

            ctx->context.push_back(next);

            // Decode token
            std::string token_text;
            if (ctx->tokenizer.is_loaded()) {
                token_text = ctx->tokenizer.decode({next});
            } else if (next >= 0 && next < 128) {
                token_text = std::string(1, static_cast<char>(next));
            }

            // Check buffer capacity
            if (generated.size() + token_text.size() + 1 > buf_len) {
                // Write what we have and return buffer-too-small
                std::memcpy(out_buf, generated.data(), generated.size());
                out_buf[generated.size()] = '\0';
                set_error("nos_generate: output buffer full, partial result returned");
                return NOS_ERR_BUFFER;
            }

            generated += token_text;

            logits = ctx->engine.forward_step(next, ctx->current_pos);
            ctx->current_pos++;
            if (logits == nullptr) break;
        }

        // Write final result
        if (generated.size() + 1 > buf_len) {
            set_error("nos_generate: output buffer too small");
            return NOS_ERR_BUFFER;
        }
        std::memcpy(out_buf, generated.data(), generated.size());
        out_buf[generated.size()] = '\0';
        return NOS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("nos_generate: ") + e.what());
        return NOS_ERR_INTERNAL;
    }
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

int nos_tokenize(nos_ctx_t* ctx, const char* text,
                 int* out_ids, size_t max_ids, size_t* num_ids) {
    clear_error();
    if (ctx == nullptr || text == nullptr || out_ids == nullptr || num_ids == nullptr) {
        set_error("nos_tokenize: null argument");
        return NOS_ERR_INVALID;
    }
    try {
        std::vector<int> ids;
        if (ctx->tokenizer.is_loaded()) {
            ids = ctx->tokenizer.encode(text);
        } else {
            const char* p = text;
            while (*p != '\0') {
                ids.push_back(
                    static_cast<int>(static_cast<unsigned char>(*p)));
                ++p;
            }
        }

        if (ids.size() > max_ids) {
            *num_ids = ids.size();
            set_error("nos_tokenize: max_ids too small, need "
                      + std::to_string(ids.size()));
            return NOS_ERR_BUFFER;
        }

        for (size_t i = 0; i < ids.size(); ++i) {
            out_ids[i] = ids[i];
        }
        *num_ids = ids.size();
        return NOS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("nos_tokenize: ") + e.what());
        return NOS_ERR_INTERNAL;
    }
}

int nos_detokenize(nos_ctx_t* ctx, const int* ids, size_t num_ids,
                   char* out_buf, size_t buf_len) {
    clear_error();
    if (ctx == nullptr || ids == nullptr || out_buf == nullptr) {
        set_error("nos_detokenize: null argument");
        return NOS_ERR_INVALID;
    }
    if (buf_len == 0) {
        set_error("nos_detokenize: buf_len is 0");
        return NOS_ERR_BUFFER;
    }
    try {
        std::vector<int> id_vec(ids, ids + num_ids);
        std::string text;
        if (ctx->tokenizer.is_loaded()) {
            text = ctx->tokenizer.decode(id_vec);
        } else {
            for (size_t i = 0; i < num_ids; ++i) {
                if (ids[i] >= 0 && ids[i] < 128) {
                    text += static_cast<char>(ids[i]);
                }
            }
        }

        if (text.size() + 1 > buf_len) {
            set_error("nos_detokenize: output buffer too small");
            return NOS_ERR_BUFFER;
        }

        std::memcpy(out_buf, text.data(), text.size());
        out_buf[text.size()] = '\0';
        return NOS_OK;
    } catch (const std::exception& e) {
        set_error(std::string("nos_detokenize: ") + e.what());
        return NOS_ERR_INTERNAL;
    }
}

// ── Metrics ─────────────────────────────────────────────────────────────────

const char* nos_get_metrics(nos_ctx_t* ctx) {
    if (ctx == nullptr) {
        return nullptr;
    }
    try {
        nlohmann::json j = ctx->engine.metrics().to_json();

        // Add TTFT
        j["ttft_ms"] = ctx->engine.ttft_ms();

        // Add routing metrics
        auto rm = ctx->engine.routing_metrics();
        j["routing"] = {
            {"total_decisions", rm.total_routing_decisions},
            {"total_switches", rm.total_switches},
            {"switch_rate", static_cast<double>(rm.switch_rate)},
            {"avg_window_length", static_cast<double>(rm.avg_window_length)}
        };

        // Add prefetch stats
        auto ps = ctx->engine.prefetch_stats();
        j["prefetch"] = {
            {"mode", ps.mode},
            {"rwp_oracle", ps.rwp_oracle},
            {"rwp_best_baseline", ps.rwp_best_baseline},
            {"effective_k", ps.effective_k},
            {"speculative_hits", ps.speculative_hits},
            {"speculative_misses", ps.speculative_misses}
        };

        // Add VMM stats
        if (ctx->vmm) {
            auto vs = ctx->vmm->stats();
            j["vmm"] = {
                {"total_pins", vs.total_pins},
                {"cache_hits", vs.cache_hits},
                {"cache_misses", vs.cache_misses},
                {"evictions", vs.evictions},
                {"resident_pages", vs.resident_pages},
                {"hit_rate", vs.hit_rate}
            };
        }

        ctx->metrics_json_cache = j.dump();
        return ctx->metrics_json_cache.c_str();
    } catch (...) {
        return nullptr;
    }
}

// ── Error Handling ──────────────────────────────────────────────────────────

const char* nos_last_error(void) {
    return tl_last_error.c_str();
}

// ── Version ─────────────────────────────────────────────────────────────────

int nos_version_major(void) { return NOS_VERSION_MAJOR; }
int nos_version_minor(void) { return NOS_VERSION_MINOR; }
int nos_version_patch(void) { return NOS_VERSION_PATCH; }

const char* nos_version(void) {
    static const char version[] = "0.1.0";
    return version;
}
