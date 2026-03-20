/// @file libneuralos.h
/// @brief NeuralOS C Library API -- stable extern "C" interface for FFI.
///
/// All functions use opaque handles and C-compatible types only.
/// Error information is available via thread-local nos_last_error().
///
/// Return codes:
///   0  = NOS_OK           -- Success
///  -1  = NOS_ERR_INVALID  -- Invalid argument (null pointer, bad handle)
///  -2  = NOS_ERR_MODEL    -- Model load / inference error
///  -3  = NOS_ERR_BUFFER   -- Buffer too small
///  -4  = NOS_ERR_INTERNAL -- Internal / unexpected error

#ifndef LIBNEURALOS_H
#define LIBNEURALOS_H

#include <stddef.h>
#include <stdint.h>

/// Version macros
#define NOS_VERSION_MAJOR 0
#define NOS_VERSION_MINOR 1
#define NOS_VERSION_PATCH 0

/// Return codes
#define NOS_OK           0
#define NOS_ERR_INVALID  (-1)
#define NOS_ERR_MODEL    (-2)
#define NOS_ERR_BUFFER   (-3)
#define NOS_ERR_INTERNAL (-4)

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque context handle.
typedef struct nos_ctx nos_ctx_t;

/// Configuration struct with ABI versioning via struct_size.
typedef struct nos_config {
    uint32_t struct_size;          ///< MUST be sizeof(nos_config_t) for ABI checks
    float    temperature;          ///< Sampling temperature (0 = greedy)
    int      top_k;                ///< Top-k filtering (0 = disabled)
    float    top_p;                ///< Top-p nucleus sampling
    float    repetition_penalty;   ///< Repetition penalty (1.0 = disabled)
    float    min_p;                ///< Min-p filtering (0.0 = disabled)
    float    sticky_lambda;        ///< Sticky routing lambda (-1 = auto)
    int      sticky_window;        ///< Max stickiness window in tokens
    int      max_seq_len;          ///< Maximum sequence length
    int      num_threads;          ///< Thread count (0 = auto)
    size_t   memory_budget;        ///< Memory budget in bytes (0 = default 8G)
    int      prefetch_enabled;     ///< Enable oracle prefetcher (0/1)
    int      prefetch_max_k;       ///< Prefetch max lookahead K
} nos_config_t;

// ── Lifecycle ─────────────────────────────────────────────────────────────

/// Create an inference context from a converted .nxp model directory.
/// @param model_path  Path to directory containing model.nxp + model_config.json
/// @param config      Configuration with struct_size set to sizeof(nos_config_t)
/// @return Context handle, or NULL on failure (check nos_last_error())
nos_ctx_t* nos_create(const char* model_path, nos_config_t config);

/// Destroy context and free all resources.
void nos_destroy(nos_ctx_t* ctx);

/// Reset KV cache and context state for a new sequence.
/// @return NOS_OK on success, error code on failure
int nos_reset(nos_ctx_t* ctx);

// ── Inference ─────────────────────────────────────────────────────────────

/// Step one token through the model.
/// @param ctx       Context handle
/// @param token_id  Input token ID
/// @param out_token Output: predicted next token ID
/// @return NOS_OK on success, error code on failure
int nos_step_token(nos_ctx_t* ctx, int token_id, int* out_token);

/// Step text through the model (tokenize, forward, decode one output token).
/// @param ctx      Context handle
/// @param text     Input text to process
/// @param out_buf  Output buffer for decoded token text
/// @param buf_len  Size of out_buf in bytes
/// @return NOS_OK on success, error code on failure
int nos_step_text(nos_ctx_t* ctx, const char* text, char* out_buf, size_t buf_len);

/// Generate text from a prompt (blocking, loops until EOS or buffer full).
/// @param ctx      Context handle
/// @param prompt   Input prompt text
/// @param out_buf  Output buffer for generated text
/// @param buf_len  Size of out_buf in bytes
/// @return NOS_OK on success, error code on failure
int nos_generate(nos_ctx_t* ctx, const char* prompt, char* out_buf, size_t buf_len);

// ── Tokenizer ─────────────────────────────────────────────────────────────

/// Tokenize text into token IDs.
/// @param ctx      Context handle
/// @param text     Input text
/// @param out_ids  Output array for token IDs
/// @param max_ids  Maximum number of IDs to write
/// @param num_ids  Output: actual number of IDs written
/// @return NOS_OK on success, NOS_ERR_BUFFER if max_ids too small
int nos_tokenize(nos_ctx_t* ctx, const char* text,
                 int* out_ids, size_t max_ids, size_t* num_ids);

/// Detokenize token IDs back to text.
/// @param ctx      Context handle
/// @param ids      Input token ID array
/// @param num_ids  Number of IDs in array
/// @param out_buf  Output buffer for decoded text
/// @param buf_len  Size of out_buf in bytes
/// @return NOS_OK on success, NOS_ERR_BUFFER if buf_len too small
int nos_detokenize(nos_ctx_t* ctx, const int* ids, size_t num_ids,
                   char* out_buf, size_t buf_len);

// ── Metrics ───────────────────────────────────────────────────────────────

/// Get metrics as a JSON string (owned by ctx, valid until next call).
/// @return JSON string, or NULL if ctx is NULL
const char* nos_get_metrics(nos_ctx_t* ctx);

// ── Error Handling ────────────────────────────────────────────────────────

/// Get the last error description for this thread.
/// @return Error string (empty if no error)
const char* nos_last_error(void);

// ── Version ───────────────────────────────────────────────────────────────

int nos_version_major(void);
int nos_version_minor(void);
int nos_version_patch(void);

/// Get version string (e.g., "0.1.0"). Statically allocated.
const char* nos_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBNEURALOS_H
