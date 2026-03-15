/// @file inference_engine.cpp
/// @brief Full transformer forward pass implementation.

#include "engine/inference_engine.h"
#include "engine/metrics.h"
#include "engine/oracle_prefetcher.h"
#include "engine/rmsnorm.h"
#include "engine/shift_detector.h"
#include "engine/sticky_router.h"
#include "engine/thread_pool.h"
#include "format/expert_format.h"
#include "kernel/bitnet_kernel.h"
#include "kernel/packing.h"
#include "vmm/vmm.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>

#include <nlohmann/json.hpp>

namespace nos {

// Reserved expert_id constants matching conversion pipeline
static constexpr uint32_t ATTN_Q_ID  = 0xFFFFFFF0u;
static constexpr uint32_t ATTN_K_ID  = 0xFFFFFFF1u;
static constexpr uint32_t ATTN_V_ID  = 0xFFFFFFF2u;
static constexpr uint32_t ATTN_O_ID  = 0xFFFFFFF3u;
static constexpr uint32_t ROUTER_ID  = 0xFFFFFFFFu;

// Minimum computation size to justify parallel dispatch overhead
static constexpr int PARALLEL_THRESHOLD = 100000;

struct ExpertWorkspace {
    std::vector<float> gate_buf;
    std::vector<float> up_buf;
    std::vector<float> ffn_out;
    std::vector<float> expert_out;
};

struct InferenceEngine::Impl {
    ModelConfig config;
    Vmm* vmm = nullptr;

    // Pre-allocated inference buffers (no allocation during forward_step)
    std::vector<float> hidden_state;     // [hidden_dim]
    std::vector<float> residual;         // [hidden_dim]
    std::vector<float> norm_out;         // [hidden_dim]
    std::vector<float> q_buf;            // [n_heads * head_dim]
    std::vector<float> k_buf;            // [n_kv_heads * head_dim]
    std::vector<float> v_buf;            // [n_kv_heads * head_dim]
    std::vector<float> attn_out;         // [n_heads * head_dim]
    std::vector<float> moe_out;          // [hidden_dim]
    std::vector<float> expert_out;       // [hidden_dim]
    std::vector<float> gate_buf;         // [intermediate_dim_per_expert]
    std::vector<float> up_buf;           // [intermediate_dim_per_expert]
    std::vector<float> ffn_out;          // [intermediate_dim_per_expert]
    std::vector<float> logits;           // [vocab_size]

    // Model weights (RAM-resident, loaded at startup)
    std::vector<uint16_t> embedding;     // [vocab_size * hidden_dim] FP16
    std::vector<uint16_t> output_proj;   // [vocab_size * hidden_dim] FP16

    // Per-layer norm weights (FP32, loaded from FP16)
    // Layout: [attn_norm_L0, ffn_norm_L0, attn_norm_L1, ..., final_norm]
    std::vector<float> norm_weights;

    // Per-layer attention weights (INT8) with FP16 scales
    struct LayerAttnWeights {
        std::vector<int8_t> wq, wk, wv, wo;
        std::vector<uint16_t> sq, sk, sv, so;
        int q_rows = 0, k_rows = 0, v_rows = 0, o_rows = 0;
    };
    std::vector<LayerAttnWeights> attn_weights;

    // Per-layer router weights (FP32, loaded from FP16)
    std::vector<std::vector<float>> router_weights;

    // Pre-loaded expert scale factors (small enough to keep in RAM)
    // Indexed by [layer][expert] -> vector of FP16 scales
    std::vector<std::vector<std::vector<uint16_t>>> expert_scales;
    int expert_rows_per_expert = 0;  // intermediate_dim / expert_count

    // Components
    RoPE rope;
    Attention attention;
    std::vector<Router> routers;

    // KV cache (from VMM)
    float* kv_cache = nullptr;
    size_t kv_cache_bytes = 0;

    // Parallel expert dispatch
    std::unique_ptr<ThreadPool> thread_pool;
    std::vector<ExpertWorkspace> workspaces;
    int num_threads = 0;

    // Sticky routing and shift detection
    StickyRouter sticky_router;
    ShiftDetector shift_detector;
    std::vector<float> prev_hidden_state;

    // Oracle prefetcher (optional, constructed in load() if prefetch_enabled)
    std::unique_ptr<OraclePrefetcher> oracle_prefetcher;
    bool prefetch_enabled = false;
    int  prefetch_max_k   = 10;

    // Metrics and TTFT tracking
    MetricsCollector metrics_collector;
    std::chrono::high_resolution_clock::time_point inference_start;
    double ttft_ms = 0.0;
    bool ttft_recorded = false;
    int token_count = 0;

    // INT8 matmul helper
    void int8_matvec(const int8_t* weights, const uint16_t* scales,
                     const float* input, float* output, int rows, int cols);

    // FP16 matmul helper (for output projection)
    void fp16_matvec(const uint16_t* weights, const float* input,
                     float* output, int rows, int cols);

    // Load attention weights for one layer
    bool load_attn_weights(NxpReader& reader, uint32_t layer,
                          LayerAttnWeights& out);

    // Load router weights for one layer
    bool load_router_weights(NxpReader& reader, uint32_t layer,
                            std::vector<float>& out);
};

void InferenceEngine::Impl::int8_matvec(const int8_t* W, const uint16_t* scales,
                                         const float* x, float* out,
                                         int rows, int cols) {
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };
    for (int r = 0; r < rows; r++) {
        float scale = fp16_to_fp32(scales[sz(r)]);
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += (static_cast<float>(W[sz(r) * sz(cols) + sz(c)]) * scale / 127.0f) * x[c];
        }
        out[r] = sum;
    }
}

void InferenceEngine::Impl::fp16_matvec(const uint16_t* W, const float* x,
                                         float* out, int rows, int cols) {
    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += fp16_to_fp32(W[sz(r) * sz(cols) + sz(c)]) * x[c];
        }
        out[r] = sum;
    }
}

bool InferenceEngine::Impl::load_attn_weights(NxpReader& reader, uint32_t layer,
                                               LayerAttnWeights& aw) {
    auto load_int8 = [&](uint32_t expert_id, std::vector<int8_t>& data,
                         std::vector<uint16_t>& scales, int& rows_out) -> bool {
        const NxpExpertEntry* entry = reader.find_expert(layer, expert_id);
        if (!entry) return false;

        data.resize(entry->size);
        if (reader.read_expert(*entry, reinterpret_cast<uint8_t*>(data.data()),
                              entry->size) < 0) {
            return false;
        }

        rows_out = static_cast<int>(entry->num_channels);
        if (rows_out > 0) {
            scales.resize(static_cast<size_t>(rows_out));
            reader.read_scales(*entry, scales.data(),
                             static_cast<size_t>(rows_out) * sizeof(uint16_t));
        }
        return true;
    };

    if (!load_int8(ATTN_Q_ID, aw.wq, aw.sq, aw.q_rows)) return false;
    if (!load_int8(ATTN_K_ID, aw.wk, aw.sk, aw.k_rows)) return false;
    if (!load_int8(ATTN_V_ID, aw.wv, aw.sv, aw.v_rows)) return false;
    if (!load_int8(ATTN_O_ID, aw.wo, aw.so, aw.o_rows)) return false;
    return true;
}

bool InferenceEngine::Impl::load_router_weights(NxpReader& reader, uint32_t layer,
                                                 std::vector<float>& out) {
    const NxpExpertEntry* entry = reader.find_expert(layer, ROUTER_ID);
    if (!entry) return false;

    // Router weights stored as FP16
    size_t fp16_count = entry->size / sizeof(uint16_t);
    std::vector<uint16_t> fp16(fp16_count);
    reader.read_expert(*entry, reinterpret_cast<uint8_t*>(fp16.data()),
                      entry->size);

    out.resize(fp16_count);
    for (size_t i = 0; i < fp16_count; i++) {
        out[i] = fp16_to_fp32(fp16[i]);
    }
    return true;
}

InferenceEngine::InferenceEngine() : impl_(std::make_unique<Impl>()) {}
InferenceEngine::~InferenceEngine() = default;

int InferenceEngine::vocab_size() const { return static_cast<int>(impl_->config.vocab_size); }

StickyRouter::AggregateMetrics InferenceEngine::routing_metrics() const {
    return impl_->sticky_router.aggregate_metrics();
}

const StickyRouter::TraceEntry& InferenceEngine::last_routing_trace() const {
    return impl_->sticky_router.last_trace();
}
const ModelConfig& InferenceEngine::config() const { return impl_->config; }

const MetricsCollector& InferenceEngine::metrics() const {
    return impl_->metrics_collector;
}

double InferenceEngine::ttft_ms() const { return impl_->ttft_ms; }

void InferenceEngine::set_sticky_config(float lambda_override, int max_window) {
    StickyRouter::Config cfg = impl_->sticky_router.config();
    cfg.lambda_override = lambda_override;
    cfg.max_window = max_window;
    impl_->sticky_router = StickyRouter(cfg);
}

bool InferenceEngine::load(const std::string& model_dir, Vmm* vmm, int num_threads) {
    auto sz = [](uint32_t v) -> size_t { return static_cast<size_t>(v); };

    impl_->vmm = vmm;

    // --- Read model_config.json ---
    std::string config_path = model_dir + "/model_config.json";
    {
        std::ifstream ifs(config_path);
        if (!ifs.is_open()) {
            std::fprintf(stderr, "ERROR: Cannot open %s\n", config_path.c_str());
            return false;
        }
        nlohmann::json j = nlohmann::json::parse(ifs);
        impl_->config = j.get<ModelConfig>();
    }

    auto& mc = impl_->config;

    // --- Initialize kernel dispatch ---
    bitnet_init();

    // --- Open .nxp file for RAM-resident weight loading ---
    std::string nxp_path = model_dir + "/model.nxp";
    NxpReader reader;
    if (!reader.open(nxp_path)) {
        std::fprintf(stderr, "ERROR: Cannot open %s\n", nxp_path.c_str());
        return false;
    }

    // --- Load embedding matrix ---
    const NxpExpertEntry* emb_entry = reader.find_expert(UINT32_MAX, 0);
    if (emb_entry) {
        impl_->embedding.resize(emb_entry->size / sizeof(uint16_t));
        reader.read_expert(*emb_entry,
                          reinterpret_cast<uint8_t*>(impl_->embedding.data()),
                          emb_entry->size);
    } else {
        // Fallback: zero embeddings for testing
        impl_->embedding.resize(sz(mc.vocab_size) * sz(mc.hidden_dim), 0);
    }

    // --- Load output projection ---
    const NxpExpertEntry* out_entry = reader.find_expert(UINT32_MAX, 1);
    if (out_entry) {
        impl_->output_proj.resize(out_entry->size / sizeof(uint16_t));
        reader.read_expert(*out_entry,
                          reinterpret_cast<uint8_t*>(impl_->output_proj.data()),
                          out_entry->size);
    } else {
        impl_->output_proj.resize(sz(mc.vocab_size) * sz(mc.hidden_dim), 0);
    }

    // --- Load RMSNorm weights ---
    // Layout: [attn_norm_L0, ffn_norm_L0, ..., final_norm] = (n_layers * 2 + 1) * hidden_dim
    const NxpExpertEntry* norm_entry = reader.find_expert(UINT32_MAX, 2);
    size_t total_norm_floats = (sz(mc.n_layers) * 2 + 1) * sz(mc.hidden_dim);
    impl_->norm_weights.resize(total_norm_floats);

    if (norm_entry) {
        size_t fp16_count = norm_entry->size / sizeof(uint16_t);
        std::vector<uint16_t> norm_fp16(fp16_count);
        reader.read_expert(*norm_entry,
                          reinterpret_cast<uint8_t*>(norm_fp16.data()),
                          norm_entry->size);
        size_t count = std::min(fp16_count, total_norm_floats);
        for (size_t i = 0; i < count; i++) {
            impl_->norm_weights[i] = fp16_to_fp32(norm_fp16[i]);
        }
    } else {
        // Default: weights of 1.0 (identity normalization)
        std::fill(impl_->norm_weights.begin(), impl_->norm_weights.end(), 1.0f);
    }

    // --- Load per-layer attention weights ---
    impl_->attn_weights.resize(sz(mc.n_layers));
    for (uint32_t L = 0; L < mc.n_layers; L++) {
        if (!impl_->load_attn_weights(reader, L, impl_->attn_weights[sz(L)])) {
            std::fprintf(stderr, "ERROR: Failed to load attention weights for layer %u\n", L);
            return false;
        }
    }

    // --- Load per-layer router weights ---
    impl_->router_weights.resize(sz(mc.n_layers));
    impl_->routers.resize(sz(mc.n_layers));
    for (uint32_t L = 0; L < mc.n_layers; L++) {
        impl_->load_router_weights(reader, L, impl_->router_weights[sz(L)]);
        if (!impl_->router_weights[sz(L)].empty()) {
            impl_->routers[sz(L)].load(
                impl_->router_weights[sz(L)].data(),
                static_cast<int>(mc.expert_count),
                static_cast<int>(mc.hidden_dim));
        }
    }

    // --- Pre-load expert scale factors ---
    impl_->expert_rows_per_expert = (mc.expert_count > 0)
        ? static_cast<int>(mc.intermediate_dim / mc.expert_count) : 0;
    if (mc.expert_count > 0) {
        impl_->expert_scales.resize(sz(mc.n_layers));
        for (uint32_t L = 0; L < mc.n_layers; L++) {
            impl_->expert_scales[sz(L)].resize(sz(mc.expert_count));
            for (uint32_t E = 0; E < mc.expert_count; E++) {
                const NxpExpertEntry* entry = reader.find_expert(L, E);
                if (entry && entry->scale_size > 0) {
                    size_t n_scales = entry->scale_size / sizeof(uint16_t);
                    impl_->expert_scales[sz(L)][sz(E)].resize(n_scales);
                    reader.read_scales(*entry,
                                      impl_->expert_scales[sz(L)][sz(E)].data(),
                                      entry->scale_size);
                }
            }
        }
    }

    reader.close();

    // --- Pre-allocate inference buffers ---
    impl_->hidden_state.resize(sz(mc.hidden_dim));
    impl_->residual.resize(sz(mc.hidden_dim));
    impl_->norm_out.resize(sz(mc.hidden_dim));
    impl_->q_buf.resize(sz(mc.n_heads) * sz(mc.head_dim));
    impl_->k_buf.resize(sz(mc.n_kv_heads) * sz(mc.head_dim));
    impl_->v_buf.resize(sz(mc.n_kv_heads) * sz(mc.head_dim));
    impl_->attn_out.resize(sz(mc.n_heads) * sz(mc.head_dim));
    impl_->moe_out.resize(sz(mc.hidden_dim));
    impl_->expert_out.resize(sz(mc.hidden_dim));
    // Expert intermediate dimension estimate
    size_t max_expert_inter = sz(mc.intermediate_dim);
    impl_->gate_buf.resize(max_expert_inter);
    impl_->up_buf.resize(max_expert_inter);
    impl_->ffn_out.resize(max_expert_inter);
    impl_->logits.resize(sz(mc.vocab_size));

    // --- Precompute RoPE tables ---
    impl_->rope.precompute(static_cast<int>(mc.head_dim),
                           static_cast<int>(mc.max_seq_len),
                           mc.rope_theta);

    // --- Initialize attention ---
    impl_->attention.init(static_cast<int>(mc.n_heads),
                          static_cast<int>(mc.n_kv_heads),
                          static_cast<int>(mc.head_dim),
                          static_cast<int>(mc.max_seq_len));

    // --- KV cache from VMM ---
    if (vmm) {
        impl_->kv_cache = static_cast<float*>(vmm->kv_cache_base());
        impl_->kv_cache_bytes = vmm->kv_cache_size();
    }
    if (!impl_->kv_cache) {
        // Fallback: self-managed KV cache
        size_t kv_per_layer = impl_->attention.kv_cache_per_layer_bytes();
        impl_->kv_cache_bytes = kv_per_layer * sz(mc.n_layers);
        // Allocate as raw bytes and store as float*
        impl_->kv_cache = reinterpret_cast<float*>(
            std::aligned_alloc(64, impl_->kv_cache_bytes));
        if (impl_->kv_cache) {
            std::memset(impl_->kv_cache, 0, impl_->kv_cache_bytes);
        }
    }

    // --- Initialize thread pool for expert-parallel dispatch ---
    int top_k_val = static_cast<int>(mc.top_k);
    if (top_k_val > 1) {
        int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (hw_threads <= 0) hw_threads = 2;
        impl_->num_threads = (num_threads > 0)
            ? num_threads
            : std::min(top_k_val, hw_threads);
        impl_->thread_pool = std::make_unique<ThreadPool>(impl_->num_threads);

        // Pre-allocate per-expert workspaces (one per top-k slot)
        size_t expert_inter = (mc.expert_count > 0)
            ? sz(mc.intermediate_dim / mc.expert_count) : 0;
        impl_->workspaces.resize(static_cast<size_t>(top_k_val));
        for (auto& ws : impl_->workspaces) {
            ws.gate_buf.resize(expert_inter);
            ws.up_buf.resize(expert_inter);
            ws.ffn_out.resize(expert_inter);
            ws.expert_out.resize(sz(mc.hidden_dim));
        }
    }

    // --- Initialize sticky routing and shift detection ---
    impl_->sticky_router.init(static_cast<int>(mc.n_layers));
    impl_->prev_hidden_state.resize(sz(mc.hidden_dim), 0.0f);

    // --- Initialize metrics ---
    impl_->metrics_collector.register_defaults();
    impl_->inference_start = std::chrono::high_resolution_clock::now();

    // --- Initialize oracle prefetcher (if enabled) ---
    if (impl_->prefetch_enabled && impl_->vmm) {
        OraclePrefetcher::Config pcfg;
        pcfg.n_layers   = static_cast<int>(mc.n_layers);
        pcfg.num_experts = static_cast<int>(mc.expert_count);
        pcfg.max_k       = impl_->prefetch_max_k;
        impl_->oracle_prefetcher = std::make_unique<OraclePrefetcher>(
            pcfg, impl_->vmm, &impl_->metrics_collector);
    }

    return true;
}

void InferenceEngine::reset_kv_cache() {
    if (impl_->kv_cache && impl_->kv_cache_bytes > 0) {
        std::memset(impl_->kv_cache, 0, impl_->kv_cache_bytes);
    }
    impl_->sticky_router.reset();
    impl_->shift_detector.reset();
    std::fill(impl_->prev_hidden_state.begin(), impl_->prev_hidden_state.end(), 0.0f);
    impl_->metrics_collector.reset();
    impl_->metrics_collector.register_defaults();
    impl_->ttft_ms = 0.0;
    impl_->ttft_recorded = false;
    impl_->token_count = 0;
    impl_->inference_start = std::chrono::high_resolution_clock::now();
    if (impl_->oracle_prefetcher) impl_->oracle_prefetcher->reset();
}

const float* InferenceEngine::forward_step(int token_id, int pos) {
    auto t_step_start = std::chrono::high_resolution_clock::now();

    auto sz = [](int v) -> size_t { return static_cast<size_t>(v); };
    auto& mc = impl_->config;
    int hidden_dim = static_cast<int>(mc.hidden_dim);
    int n_heads = static_cast<int>(mc.n_heads);
    int n_kv_heads = static_cast<int>(mc.n_kv_heads);
    int head_dim = static_cast<int>(mc.head_dim);
    int n_layers = static_cast<int>(mc.n_layers);
    int top_k = static_cast<int>(mc.top_k);
    float eps = mc.norm_eps;

    // 1. Embedding lookup: FP16 -> FP32
    size_t emb_offset = sz(token_id) * sz(hidden_dim);
    for (int d = 0; d < hidden_dim; d++) {
        impl_->hidden_state[sz(d)] = fp16_to_fp32(
            impl_->embedding[emb_offset + sz(d)]);
    }

    // KV cache layout per layer
    size_t kv_per_layer = impl_->attention.kv_cache_per_layer_bytes() / sizeof(float);

    // 2. For each layer
    for (int L = 0; L < n_layers; L++) {
        // 2a. Save residual
        std::memcpy(impl_->residual.data(), impl_->hidden_state.data(),
                    sz(hidden_dim) * sizeof(float));

        // 2b. Attention norm
        size_t norm_idx = sz(L) * 2 * sz(hidden_dim);
        rmsnorm(impl_->norm_out.data(), impl_->hidden_state.data(),
                impl_->norm_weights.data() + norm_idx, hidden_dim, eps);

        // 2c. INT8 attention projections
        auto& aw = impl_->attn_weights[sz(L)];
        if (!aw.wq.empty()) {
            int q_cols = static_cast<int>(aw.wq.size()) / aw.q_rows;
            int k_cols = static_cast<int>(aw.wk.size()) / aw.k_rows;
            int v_cols = static_cast<int>(aw.wv.size()) / aw.v_rows;

            impl_->int8_matvec(aw.wq.data(), aw.sq.data(),
                              impl_->norm_out.data(), impl_->q_buf.data(),
                              aw.q_rows, q_cols);
            impl_->int8_matvec(aw.wk.data(), aw.sk.data(),
                              impl_->norm_out.data(), impl_->k_buf.data(),
                              aw.k_rows, k_cols);
            impl_->int8_matvec(aw.wv.data(), aw.sv.data(),
                              impl_->norm_out.data(), impl_->v_buf.data(),
                              aw.v_rows, v_cols);
        }

        // 2d. RoPE
        impl_->rope.apply(impl_->q_buf.data(), impl_->k_buf.data(),
                          n_heads, n_kv_heads, head_dim, pos);

        // 2e. Attention
        float* layer_kv = impl_->kv_cache + sz(L) * kv_per_layer;
        impl_->attention.forward(impl_->attn_out.data(),
                                impl_->q_buf.data(), impl_->k_buf.data(),
                                impl_->v_buf.data(), pos, layer_kv);

        // 2f. Output projection
        if (!aw.wo.empty()) {
            int o_cols = static_cast<int>(aw.wo.size()) / aw.o_rows;
            impl_->int8_matvec(aw.wo.data(), aw.so.data(),
                              impl_->attn_out.data(), impl_->hidden_state.data(),
                              aw.o_rows, o_cols);
        } else {
            std::memcpy(impl_->hidden_state.data(), impl_->attn_out.data(),
                        sz(hidden_dim) * sizeof(float));
        }

        // 2g. Residual connection
        for (int d = 0; d < hidden_dim; d++) {
            impl_->hidden_state[sz(d)] += impl_->residual[sz(d)];
        }

        // 2h. Save residual for FFN
        std::memcpy(impl_->residual.data(), impl_->hidden_state.data(),
                    sz(hidden_dim) * sizeof(float));

        // 2i. FFN norm
        size_t ffn_norm_idx = (sz(L) * 2 + 1) * sz(hidden_dim);
        rmsnorm(impl_->norm_out.data(), impl_->hidden_state.data(),
                impl_->norm_weights.data() + ffn_norm_idx, hidden_dim, eps);

        // 2j. MoE routing with sticky routing and shift detection
        if (mc.expert_count > 0 && !impl_->router_weights[sz(L)].empty()) {
            // Compute shift detection signals
            float attn_entropy = impl_->attention.last_entropy();
            float cos_sim = cosine_similarity(
                impl_->hidden_state.data(), impl_->prev_hidden_state.data(),
                hidden_dim);

            // Get base router result (includes raw_scores for variance)
            RouterResult base_route = impl_->routers[sz(L)].route(
                impl_->norm_out.data(), top_k);

            // Compute router logit variance from raw_scores
            float logit_var = 0.0f;
            if (!base_route.raw_scores.empty()) {
                float mean = 0.0f;
                for (float s : base_route.raw_scores) mean += s;
                mean /= static_cast<float>(base_route.raw_scores.size());
                float sq_sum = 0.0f;
                for (float s : base_route.raw_scores) {
                    float d = s - mean;
                    sq_sum += d * d;
                }
                logit_var = sq_sum / static_cast<float>(base_route.raw_scores.size());
            }

            // Evaluate shift detector
            auto shift = impl_->shift_detector.evaluate(
                attn_entropy, cos_sim, logit_var);

            // Track shift detection
            if (shift.detected) {
                impl_->metrics_collector.inc_counter("shift_detections");
            }

            // Route via sticky router (wraps base router with stickiness)
            RouterResult route = impl_->sticky_router.route(
                impl_->routers[sz(L)], impl_->norm_out.data(), top_k,
                static_cast<uint32_t>(L), pos, shift.detected);

            // Track routing decisions
            impl_->metrics_collector.inc_counter("total_routing_decisions");
            if (impl_->sticky_router.last_trace().switched) {
                impl_->metrics_collector.inc_counter("expert_switches");
            }

            // 2j'. Oracle prefetch: replaces naive 1-layer lookahead with
            // variable-K speculative prefetch via OraclePrefetcher.
            if (impl_->oracle_prefetcher) {
                const float* raw_logits = route.raw_scores.empty()
                    ? nullptr : route.raw_scores.data();
                impl_->oracle_prefetcher->predict_and_dispatch(
                    L,
                    route.expert_ids,
                    raw_logits,             // router_logits (may be null; handled in oracle)
                    impl_->norm_out.data(), // hidden_state proxy
                    hidden_dim,
                    top_k);
            }

            // 2k. Zero MoE output
            std::memset(impl_->moe_out.data(), 0, sz(hidden_dim) * sizeof(float));

            int expert_rows = impl_->expert_rows_per_expert;
            int packed_cols = (hidden_dim + 4) / 5;
            bool use_parallel = impl_->thread_pool && top_k > 1
                && static_cast<long>(expert_rows) * hidden_dim > PARALLEL_THRESHOLD;

            if (use_parallel) {
                // 2l. Pin all experts (may already be prefetched from previous layer)
                struct PinnedExpert {
                    ExpertHandle handle;
                    const uint8_t* data;
                };
                std::vector<PinnedExpert> pinned(route.expert_ids.size());
                auto io_before = impl_->vmm->stats();
                for (size_t ei = 0; ei < route.expert_ids.size(); ei++) {
                    auto t_pin_start = std::chrono::high_resolution_clock::now();
                    pinned[ei].handle = impl_->vmm->get_handle(
                        static_cast<uint32_t>(L), route.expert_ids[ei]);
                    pinned[ei].data = impl_->vmm->pin(pinned[ei].handle);
                    auto t_pin_end = std::chrono::high_resolution_clock::now();
                    double pin_us = std::chrono::duration<double, std::micro>(
                        t_pin_end - t_pin_start).count();
                    impl_->metrics_collector.observe_histogram("io_latency_us", pin_us);
                }
                auto io_after = impl_->vmm->stats();
                uint64_t batch_hits = io_after.cache_hits - io_before.cache_hits;
                uint64_t batch_misses = io_after.cache_misses - io_before.cache_misses;
                impl_->metrics_collector.inc_counter("cache_hits", batch_hits);
                impl_->metrics_collector.inc_counter("cache_misses", batch_misses);
                impl_->metrics_collector.inc_counter("expert_loads", batch_misses);
                impl_->metrics_collector.inc_counter("expert_reuses", batch_hits);

                // 2l'. Parallel dispatch: one task per expert, each writes
                // to its own ExpertWorkspace (no sharing, no locks needed).
                std::vector<std::function<void()>> tasks;
                tasks.reserve(route.expert_ids.size());
                for (size_t ei = 0; ei < route.expert_ids.size(); ei++) {
                    if (!pinned[ei].data) continue;
                    auto* ws = &impl_->workspaces[ei];
                    const uint8_t* edata = pinned[ei].data;
                    uint32_t eid = route.expert_ids[ei];
                    const uint16_t* scales =
                        impl_->expert_scales[sz(L)][static_cast<size_t>(eid)].data();
                    const float* input = impl_->norm_out.data();

                    tasks.push_back([ws, edata, scales, input,
                                     expert_rows, hidden_dim, packed_cols]() {
                        auto usz = [](int v) -> size_t {
                            return static_cast<size_t>(v);
                        };
                        const uint8_t* pw = edata;

                        // Gate projection
                        bitnet_matvec(pw, input, ws->gate_buf.data(),
                                     expert_rows, hidden_dim, scales);

                        // Up projection
                        const uint8_t* up_pw = pw
                            + usz(expert_rows) * usz(packed_cols);
                        const uint16_t* up_sc = scales + usz(expert_rows);
                        bitnet_matvec(up_pw, input, ws->up_buf.data(),
                                     expert_rows, hidden_dim, up_sc);

                        // SwiGLU activation
                        for (int i = 0; i < expert_rows; i++) {
                            float g = ws->gate_buf[usz(i)];
                            float silu = g / (1.0f + std::exp(-g));
                            ws->ffn_out[usz(i)] = silu * ws->up_buf[usz(i)];
                        }

                        // Down projection
                        const uint8_t* down_pw = pw
                            + usz(expert_rows) * 2 * usz(packed_cols);
                        const uint16_t* down_sc = scales
                            + usz(expert_rows) * 2;
                        bitnet_matvec(down_pw, ws->ffn_out.data(),
                                     ws->expert_out.data(),
                                     hidden_dim, expert_rows, down_sc);
                    });
                }

                std::span<std::function<void()>> task_span(tasks);
                impl_->thread_pool->dispatch_batch(task_span);

                // 2l''. Combine gated results (single-threaded, after barrier)
                for (size_t ei = 0; ei < route.expert_ids.size(); ei++) {
                    if (!pinned[ei].data) continue;
                    float gate = route.gates[ei];
                    const auto& ws = impl_->workspaces[ei];
                    for (int d = 0; d < hidden_dim; d++) {
                        impl_->moe_out[sz(d)] += gate * ws.expert_out[sz(d)];
                    }
                }

                // 2l'''. Unpin all experts
                for (auto& pe : pinned) {
                    if (pe.data) impl_->vmm->unpin(pe.handle);
                }
            } else {
                // Sequential fallback (top_k=1, small model, or no pool)
                for (size_t ei = 0; ei < route.expert_ids.size(); ei++) {
                    uint32_t expert_id = route.expert_ids[ei];
                    float gate = route.gates[ei];

                    auto seq_io_before = impl_->vmm->stats();
                    auto t_pin_start = std::chrono::high_resolution_clock::now();
                    ExpertHandle handle = impl_->vmm->get_handle(
                        static_cast<uint32_t>(L), expert_id);

                    const uint8_t* expert_data = impl_->vmm->pin(handle);
                    auto t_pin_end = std::chrono::high_resolution_clock::now();
                    double pin_us = std::chrono::duration<double, std::micro>(
                        t_pin_end - t_pin_start).count();
                    impl_->metrics_collector.observe_histogram("io_latency_us", pin_us);

                    auto seq_io_after = impl_->vmm->stats();
                    if (seq_io_after.cache_hits > seq_io_before.cache_hits) {
                        impl_->metrics_collector.inc_counter("cache_hits");
                        impl_->metrics_collector.inc_counter("expert_reuses");
                    } else if (seq_io_after.cache_misses > seq_io_before.cache_misses) {
                        impl_->metrics_collector.inc_counter("cache_misses");
                        impl_->metrics_collector.inc_counter("expert_loads");
                    }
                    if (!expert_data) continue;

                    const uint8_t* packed_weights = expert_data;
                    const uint16_t* scale_factors =
                        impl_->expert_scales[sz(L)][static_cast<size_t>(expert_id)].data();

                    bitnet_matvec(packed_weights,
                                 impl_->norm_out.data(), impl_->gate_buf.data(),
                                 expert_rows, hidden_dim, scale_factors);

                    const uint8_t* up_packed = packed_weights
                        + sz(expert_rows) * sz(packed_cols);
                    const uint16_t* up_scales = scale_factors + sz(expert_rows);
                    bitnet_matvec(up_packed,
                                 impl_->norm_out.data(), impl_->up_buf.data(),
                                 expert_rows, hidden_dim, up_scales);

                    for (int i = 0; i < expert_rows; i++) {
                        float g = impl_->gate_buf[sz(i)];
                        float silu = g / (1.0f + std::exp(-g));
                        impl_->ffn_out[sz(i)] = silu * impl_->up_buf[sz(i)];
                    }

                    const uint8_t* down_packed = packed_weights
                        + sz(expert_rows) * 2 * sz(packed_cols);
                    const uint16_t* down_scales = scale_factors
                        + sz(expert_rows) * 2;
                    bitnet_matvec(down_packed,
                                 impl_->ffn_out.data(), impl_->expert_out.data(),
                                 hidden_dim, expert_rows, down_scales);

                    for (int d = 0; d < hidden_dim; d++) {
                        impl_->moe_out[sz(d)] += gate * impl_->expert_out[sz(d)];
                    }

                    impl_->vmm->unpin(handle);
                }
            }

            // 2m. Residual + MoE output
            for (int d = 0; d < hidden_dim; d++) {
                impl_->hidden_state[sz(d)] = impl_->residual[sz(d)]
                                            + impl_->moe_out[sz(d)];
            }
        } else {
            // No MoE: identity (hidden_state stays as residual)
            std::memcpy(impl_->hidden_state.data(), impl_->residual.data(),
                        sz(hidden_dim) * sizeof(float));
        }
    }

    // 2n. Per-token sticky routing feedback
    if (impl_->vmm) {
        VmmStats vs = impl_->vmm->stats();
        float miss_rate = (vs.total_pins > 0)
            ? static_cast<float>(vs.cache_misses) / static_cast<float>(vs.total_pins)
            : 0.0f;
        impl_->sticky_router.update_io_pressure(miss_rate);
    }
    // Store current hidden state for next token's cosine similarity
    std::memcpy(impl_->prev_hidden_state.data(), impl_->hidden_state.data(),
                sz(hidden_dim) * sizeof(float));

    // 3. Final RMSNorm
    size_t final_norm_idx = sz(n_layers) * 2 * sz(hidden_dim);
    rmsnorm(impl_->norm_out.data(), impl_->hidden_state.data(),
            impl_->norm_weights.data() + final_norm_idx, hidden_dim, eps);

    // 4. Output projection: FP16 weights -> FP32 logits
    impl_->fp16_matvec(impl_->output_proj.data(), impl_->norm_out.data(),
                       impl_->logits.data(),
                       static_cast<int>(mc.vocab_size), hidden_dim);

    // --- Metrics instrumentation ---
    impl_->token_count++;
    impl_->metrics_collector.inc_counter("tokens_generated");

    // Per-token latency
    auto t_step_end = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(
        t_step_end - t_step_start).count();
    impl_->metrics_collector.observe_histogram("token_latency_ms", step_ms);

    // TTFT: record time of first token
    if (!impl_->ttft_recorded) {
        impl_->ttft_ms = std::chrono::duration<double, std::milli>(
            t_step_end - impl_->inference_start).count();
        impl_->ttft_recorded = true;
    }

    // Periodic timeline recording (every 10 tokens)
    if (impl_->token_count % 10 == 0 && impl_->vmm) {
        double ts = std::chrono::duration<double>(
            t_step_end - impl_->inference_start).count();
        VmmStats vs = impl_->vmm->stats();
        float miss_rate = (vs.total_pins > 0)
            ? static_cast<float>(vs.cache_misses) / static_cast<float>(vs.total_pins)
            : 0.0f;
        impl_->metrics_collector.record_timeline(
            "io_pressure", ts, static_cast<double>(miss_rate));
        // Memory utilization: resident pages as fraction of total capacity
        double resident_ratio = (vs.total_pins > 0)
            ? static_cast<double>(vs.cache_hits) / static_cast<double>(vs.total_pins)
            : 0.0;
        impl_->metrics_collector.record_timeline(
            "memory_utilization", ts, resident_ratio);
    }

    // Tick oracle prefetcher once per token (after all layers done)
    if (impl_->oracle_prefetcher) {
        double tok_per_sec = 0.0;
        auto tl = impl_->metrics_collector.get_timeline("tok_per_sec");
        if (tl.points.size() >= 2) {
            auto& back = tl.points.back();
            auto& prev = tl.points[tl.points.size() - 2];
            if (back.first > prev.first) {
                tok_per_sec = 1000.0 / (back.first - prev.first);
            }
        }
        impl_->oracle_prefetcher->tick(tok_per_sec);
    }

    return impl_->logits.data();
}

void InferenceEngine::set_prefetch_config(bool enabled, int max_k) {
    impl_->prefetch_enabled = enabled;
    impl_->prefetch_max_k   = max_k;
}

PrefetchStats InferenceEngine::prefetch_stats() const {
    if (impl_->oracle_prefetcher) {
        return impl_->oracle_prefetcher->stats();
    }
    return PrefetchStats{};
}

}  // namespace nos
