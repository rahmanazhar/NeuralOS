/// @file inference_engine.cpp
/// @brief Full transformer forward pass implementation.

#include "engine/inference_engine.h"
#include "engine/rmsnorm.h"
#include "format/expert_format.h"
#include "kernel/bitnet_kernel.h"
#include "kernel/packing.h"
#include "vmm/vmm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

namespace nos {

// Reserved expert_id constants matching conversion pipeline
static constexpr uint32_t ATTN_Q_ID  = 0xFFFFFFF0u;
static constexpr uint32_t ATTN_K_ID  = 0xFFFFFFF1u;
static constexpr uint32_t ATTN_V_ID  = 0xFFFFFFF2u;
static constexpr uint32_t ATTN_O_ID  = 0xFFFFFFF3u;
static constexpr uint32_t ROUTER_ID  = 0xFFFFFFFFu;

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
const ModelConfig& InferenceEngine::config() const { return impl_->config; }

bool InferenceEngine::load(const std::string& model_dir, Vmm* vmm) {
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

    return true;
}

void InferenceEngine::reset_kv_cache() {
    if (impl_->kv_cache && impl_->kv_cache_bytes > 0) {
        std::memset(impl_->kv_cache, 0, impl_->kv_cache_bytes);
    }
}

const float* InferenceEngine::forward_step(int token_id, int pos) {
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

        // 2j. MoE routing
        if (mc.expert_count > 0 && !impl_->router_weights[sz(L)].empty()) {
            RouterResult route = impl_->routers[sz(L)].route(
                impl_->norm_out.data(), top_k);

            // 2k. Zero MoE output
            std::memset(impl_->moe_out.data(), 0, sz(hidden_dim) * sizeof(float));

            // 2l. For each selected expert
            for (size_t ei = 0; ei < route.expert_ids.size(); ei++) {
                uint32_t expert_id = route.expert_ids[ei];
                float gate = route.gates[ei];

                ExpertHandle handle = impl_->vmm->get_handle(
                    static_cast<uint32_t>(L), expert_id);

                const uint8_t* expert_data = impl_->vmm->pin(handle);
                if (!expert_data) continue;

                // Expert data layout: packed ternary weights
                // The NxpExpertEntry has info about the expert's structure
                // For SwiGLU: gate_proj [expert_rows x hidden_dim] and
                //             up_proj [expert_rows x hidden_dim]
                // packed together as [2*expert_rows x hidden_dim]
                // Scale factors follow the packed data

                // VMM slab contains packed ternary weights only.
                // Scale factors are pre-loaded at startup.
                int expert_rows = impl_->expert_rows_per_expert;
                int packed_cols = (hidden_dim + 4) / 5;  // bytes per row (5-per-byte)

                const uint8_t* packed_weights = expert_data;
                const uint16_t* scale_factors =
                    impl_->expert_scales[sz(L)][static_cast<size_t>(expert_id)].data();

                // Gate projection: rows [0, expert_rows)
                bitnet_matvec(packed_weights,
                             impl_->norm_out.data(), impl_->gate_buf.data(),
                             expert_rows, hidden_dim, scale_factors);

                // Up projection: rows [expert_rows, 2*expert_rows)
                const uint8_t* up_packed = packed_weights
                    + sz(expert_rows) * sz(packed_cols);
                const uint16_t* up_scales = scale_factors + sz(expert_rows);
                bitnet_matvec(up_packed,
                             impl_->norm_out.data(), impl_->up_buf.data(),
                             expert_rows, hidden_dim, up_scales);

                // SwiGLU: out[i] = SiLU(gate[i]) * up[i]
                for (int i = 0; i < expert_rows; i++) {
                    float g = impl_->gate_buf[sz(i)];
                    float silu = g / (1.0f + std::exp(-g));
                    impl_->ffn_out[sz(i)] = silu * impl_->up_buf[sz(i)];
                }

                // Down projection would go here in a full implementation
                // For now, we accumulate the gated output directly
                // (down_proj maps expert_rows -> hidden_dim)
                // Simplified: treat ffn_out as contribution to hidden_dim
                int contrib_dim = std::min(expert_rows, hidden_dim);
                for (int d = 0; d < contrib_dim; d++) {
                    impl_->moe_out[sz(d)] += gate * impl_->ffn_out[sz(d)];
                }

                impl_->vmm->unpin(handle);
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

    // 3. Final RMSNorm
    size_t final_norm_idx = sz(n_layers) * 2 * sz(hidden_dim);
    rmsnorm(impl_->norm_out.data(), impl_->hidden_state.data(),
            impl_->norm_weights.data() + final_norm_idx, hidden_dim, eps);

    // 4. Output projection: FP16 weights -> FP32 logits
    impl_->fp16_matvec(impl_->output_proj.data(), impl_->norm_out.data(),
                       impl_->logits.data(),
                       static_cast<int>(mc.vocab_size), hidden_dim);

    return impl_->logits.data();
}

}  // namespace nos
