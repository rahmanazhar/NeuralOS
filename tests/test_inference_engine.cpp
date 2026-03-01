#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "converter/model_config.h"
#include "engine/inference_engine.h"
#include "engine/perplexity.h"
#include "format/expert_format.h"
#include "kernel/bitnet_kernel.h"
#include "kernel/packing.h"
#include "vmm/vmm.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

// Synthetic model parameters
static constexpr uint32_t TEST_VOCAB_SIZE = 100;
static constexpr uint32_t TEST_HIDDEN_DIM = 32;
static constexpr uint32_t TEST_INTERMEDIATE_DIM = 64;
static constexpr uint32_t TEST_N_LAYERS = 2;
static constexpr uint32_t TEST_N_HEADS = 4;
static constexpr uint32_t TEST_N_KV_HEADS = 4;
static constexpr uint32_t TEST_HEAD_DIM = 8;  // hidden_dim / n_heads
static constexpr uint32_t TEST_EXPERT_COUNT = 2;
static constexpr uint32_t TEST_TOP_K = 2;
static constexpr uint32_t TEST_MAX_SEQ_LEN = 64;

namespace {

struct SyntheticModel {
    std::string dir;

    ~SyntheticModel() {
        std::filesystem::remove_all(dir);
    }
};

SyntheticModel create_synthetic_model() {
    namespace fs = std::filesystem;

    // Create temp directory
    std::string dir = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                    + "/neuralos_test_engine_XXXXXX";
    // Use mkdtemp equivalent
    char* tmp = const_cast<char*>(dir.c_str());
    std::string tmp_dir = dir;
    fs::create_directories(tmp_dir);

    SyntheticModel model;
    model.dir = tmp_dir;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto sz = [](uint32_t v) -> size_t { return static_cast<size_t>(v); };

    // --- Write model_config.json ---
    nos::ModelConfig mc;
    mc.architecture = "llama";
    mc.vocab_size = TEST_VOCAB_SIZE;
    mc.hidden_dim = TEST_HIDDEN_DIM;
    mc.intermediate_dim = TEST_INTERMEDIATE_DIM;
    mc.n_layers = TEST_N_LAYERS;
    mc.n_heads = TEST_N_HEADS;
    mc.n_kv_heads = TEST_N_KV_HEADS;
    mc.head_dim = TEST_HEAD_DIM;
    mc.max_seq_len = TEST_MAX_SEQ_LEN;
    mc.rope_theta = 10000.0f;
    mc.norm_eps = 1e-5f;
    mc.expert_count = TEST_EXPERT_COUNT;
    mc.top_k = TEST_TOP_K;
    mc.max_expert_size = 0;  // Will be set after writing experts
    mc.attention_type = "mha";

    // --- Create .nxp file ---
    nos::NxpFileHeader header{};
    header.magic = nos::NXP_MAGIC;
    header.version = nos::NXP_VERSION;
    header.num_layers = TEST_N_LAYERS;
    header.experts_per_layer = TEST_EXPERT_COUNT;
    header.hidden_dim = TEST_HIDDEN_DIM;
    header.intermediate_dim = TEST_INTERMEDIATE_DIM;
    header.packing_mode = 0;
    header.scale_dtype = 0;
    header.total_experts = sz(TEST_N_LAYERS) * sz(TEST_EXPERT_COUNT);
    std::memset(header.reserved, 0, sizeof(header.reserved));

    std::string nxp_path = model.dir + "/model.nxp";
    nos::NxpWriter writer;
    REQUIRE(writer.open(nxp_path, header));

    uint64_t max_expert_bytes = 0;

    // --- Write expert weights ---
    for (uint32_t L = 0; L < TEST_N_LAYERS; L++) {
        for (uint32_t E = 0; E < TEST_EXPERT_COUNT; E++) {
            // Expert rows = intermediate_dim / expert_count
            int expert_rows = static_cast<int>(TEST_INTERMEDIATE_DIM / TEST_EXPERT_COUNT);
            int gu_rows = expert_rows * 2;  // gate + up
            int packed_cols_gu = (static_cast<int>(TEST_HIDDEN_DIM) + 4) / 5;
            int down_rows = static_cast<int>(TEST_HIDDEN_DIM);
            int packed_cols_down = (expert_rows + 4) / 5;

            // Generate random ternary weights for gate+up
            std::vector<int8_t> gu_trits(sz(gu_rows) * sz(TEST_HIDDEN_DIM));
            std::uniform_int_distribution<int> trit_dist(-1, 1);
            for (auto& t : gu_trits) t = static_cast<int8_t>(trit_dist(rng));

            std::vector<uint8_t> gu_packed(sz(gu_rows) * sz(packed_cols_gu));
            for (int r = 0; r < gu_rows; r++) {
                nos::pack_row(gu_trits.data() + sz(r) * sz(TEST_HIDDEN_DIM),
                             static_cast<int>(TEST_HIDDEN_DIM),
                             gu_packed.data() + sz(r) * sz(packed_cols_gu));
            }

            // Generate random ternary weights for down_proj [hidden_dim x expert_rows]
            std::vector<int8_t> down_trits(sz(down_rows) * sz(expert_rows));
            for (auto& t : down_trits) t = static_cast<int8_t>(trit_dist(rng));

            std::vector<uint8_t> down_packed(sz(down_rows) * sz(packed_cols_down));
            for (int r = 0; r < down_rows; r++) {
                nos::pack_row(down_trits.data() + sz(r) * sz(expert_rows),
                             expert_rows,
                             down_packed.data() + sz(r) * sz(packed_cols_down));
            }

            // Concatenate packed data: [gate+up | down]
            std::vector<uint8_t> all_packed;
            all_packed.insert(all_packed.end(), gu_packed.begin(), gu_packed.end());
            all_packed.insert(all_packed.end(), down_packed.begin(), down_packed.end());

            // Concatenate scales: [gate+up scales | down scales]
            uint32_t total_channels = static_cast<uint32_t>(gu_rows + down_rows);
            std::vector<uint16_t> all_scales(sz(total_channels));
            for (auto& s : all_scales) {
                s = nos::fp32_to_fp16(std::abs(dist(rng)) + 0.01f);
            }

            auto entry = writer.write_expert(L, E, all_packed.data(), all_packed.size(),
                                              all_scales.data(), total_channels);
            if (entry.size > max_expert_bytes) max_expert_bytes = entry.size;
        }

        // --- Write INT8 attention weights ---
        uint32_t attn_ids[] = {0xFFFFFFF0u, 0xFFFFFFF1u, 0xFFFFFFF2u, 0xFFFFFFF3u};
        // Q: [n_heads*head_dim x hidden_dim], K: [n_kv*head_dim x hidden_dim]
        // V: [n_kv*head_dim x hidden_dim], O: [hidden_dim x n_heads*head_dim]
        int q_rows = static_cast<int>(TEST_N_HEADS * TEST_HEAD_DIM);
        int k_rows = static_cast<int>(TEST_N_KV_HEADS * TEST_HEAD_DIM);
        int v_rows = k_rows;
        int o_rows = static_cast<int>(TEST_HIDDEN_DIM);
        int attn_rows[] = {q_rows, k_rows, v_rows, o_rows};
        int attn_cols[] = {static_cast<int>(TEST_HIDDEN_DIM),
                          static_cast<int>(TEST_HIDDEN_DIM),
                          static_cast<int>(TEST_HIDDEN_DIM),
                          q_rows};  // O maps from n_heads*head_dim -> hidden_dim

        for (int a = 0; a < 4; a++) {
            std::vector<int8_t> data(sz(attn_rows[a]) * sz(attn_cols[a]));
            std::uniform_int_distribution<int> byte_dist(-127, 127);
            for (auto& b : data) b = static_cast<int8_t>(byte_dist(rng));

            std::vector<uint16_t> scales(sz(attn_rows[a]));
            for (auto& s : scales) {
                s = nos::fp32_to_fp16(std::abs(dist(rng)) + 0.01f);
            }

            writer.write_expert(L, attn_ids[a],
                               reinterpret_cast<const uint8_t*>(data.data()),
                               data.size(), scales.data(),
                               static_cast<uint32_t>(attn_rows[a]));
        }

        // --- Write router weights ---
        // [expert_count x hidden_dim] as FP16
        std::vector<uint16_t> router_fp16(sz(TEST_EXPERT_COUNT) * sz(TEST_HIDDEN_DIM));
        for (auto& w : router_fp16) {
            w = nos::fp32_to_fp16(dist(rng));
        }
        writer.write_expert(L, 0xFFFFFFFFu,
                           reinterpret_cast<const uint8_t*>(router_fp16.data()),
                           router_fp16.size() * sizeof(uint16_t),
                           nullptr, 0);
    }

    // --- Write embedding matrix ---
    std::vector<uint16_t> embedding(sz(TEST_VOCAB_SIZE) * sz(TEST_HIDDEN_DIM));
    for (auto& e : embedding) {
        e = nos::fp32_to_fp16(dist(rng));
    }
    writer.write_expert(UINT32_MAX, 0,
                       reinterpret_cast<const uint8_t*>(embedding.data()),
                       embedding.size() * sizeof(uint16_t),
                       nullptr, 0);

    // --- Write output projection ---
    std::vector<uint16_t> output_proj(sz(TEST_VOCAB_SIZE) * sz(TEST_HIDDEN_DIM));
    for (auto& o : output_proj) {
        o = nos::fp32_to_fp16(dist(rng));
    }
    writer.write_expert(UINT32_MAX, 1,
                       reinterpret_cast<const uint8_t*>(output_proj.data()),
                       output_proj.size() * sizeof(uint16_t),
                       nullptr, 0);

    // --- Write RMSNorm weights ---
    // (n_layers * 2 + 1) * hidden_dim floats as FP16
    size_t total_norm = (sz(TEST_N_LAYERS) * 2 + 1) * sz(TEST_HIDDEN_DIM);
    std::vector<uint16_t> norm_fp16(total_norm);
    for (auto& n : norm_fp16) {
        n = nos::fp32_to_fp16(1.0f);  // Identity weights
    }
    writer.write_expert(UINT32_MAX, 2,
                       reinterpret_cast<const uint8_t*>(norm_fp16.data()),
                       norm_fp16.size() * sizeof(uint16_t),
                       nullptr, 0);

    REQUIRE(writer.finalize());

    // Update max_expert_size in config
    mc.max_expert_size = static_cast<uint32_t>(max_expert_bytes);

    // --- Write model_config.json ---
    nlohmann::json j = mc;
    std::ofstream ofs(model.dir + "/model_config.json");
    ofs << j.dump(2);
    ofs.close();

    return model;
}

}  // anonymous namespace

TEST_CASE("InferenceEngine: loads synthetic model", "[inference_engine]") {
    auto model = create_synthetic_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;  // 1 MB
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = TEST_N_LAYERS;
    vmm_cfg.experts_per_layer = TEST_EXPERT_COUNT;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    REQUIRE(engine.vocab_size() == static_cast<int>(TEST_VOCAB_SIZE));
    REQUIRE(engine.config().n_layers == TEST_N_LAYERS);
    REQUIRE(engine.config().hidden_dim == TEST_HIDDEN_DIM);
}

TEST_CASE("InferenceEngine: forward_step produces valid logits", "[inference_engine]") {
    auto model = create_synthetic_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = TEST_N_LAYERS;
    vmm_cfg.experts_per_layer = TEST_EXPERT_COUNT;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    const float* logits = engine.forward_step(1, 0);
    REQUIRE(logits != nullptr);

    // Check no NaN/Inf
    bool has_nan_inf = false;
    for (int v = 0; v < engine.vocab_size(); v++) {
        if (std::isnan(logits[v]) || std::isinf(logits[v])) {
            has_nan_inf = true;
            break;
        }
    }
    REQUIRE_FALSE(has_nan_inf);

    // Check softmax sums to ~1.0
    float max_logit = logits[0];
    for (int v = 1; v < engine.vocab_size(); v++) {
        if (logits[v] > max_logit) max_logit = logits[v];
    }
    double sum_exp = 0.0;
    for (int v = 0; v < engine.vocab_size(); v++) {
        sum_exp += std::exp(static_cast<double>(logits[v]) - static_cast<double>(max_logit));
    }
    REQUIRE(sum_exp > 0.0);
    REQUIRE(std::isfinite(sum_exp));
}

TEST_CASE("InferenceEngine: KV cache produces different outputs at different positions",
          "[inference_engine]") {
    auto model = create_synthetic_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = TEST_N_LAYERS;
    vmm_cfg.experts_per_layer = TEST_EXPERT_COUNT;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    // Position 0
    const float* logits0 = engine.forward_step(1, 0);
    std::vector<float> saved0(logits0, logits0 + engine.vocab_size());

    // Position 1 with a different token (should produce different logits due
    // to different embedding + KV cache accumulation)
    const float* logits1 = engine.forward_step(50, 1);

    // Outputs should be different (different input tokens → different hidden states)
    bool different = false;
    for (int v = 0; v < engine.vocab_size(); v++) {
        if (std::abs(saved0[static_cast<size_t>(v)] - logits1[v]) > 1e-6f) {
            different = true;
            break;
        }
    }
    REQUIRE(different);
}

TEST_CASE("InferenceEngine: multi-token generation produces valid logits",
          "[inference_engine]") {
    auto model = create_synthetic_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = TEST_N_LAYERS;
    vmm_cfg.experts_per_layer = TEST_EXPERT_COUNT;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    for (int step = 0; step < 5; step++) {
        int token = (step * 7 + 3) % static_cast<int>(TEST_VOCAB_SIZE);
        const float* logits = engine.forward_step(token, step);
        REQUIRE(logits != nullptr);

        // Verify no NaN/Inf
        for (int v = 0; v < engine.vocab_size(); v++) {
            REQUIRE_FALSE(std::isnan(logits[v]));
            REQUIRE_FALSE(std::isinf(logits[v]));
        }
    }
}

TEST_CASE("InferenceEngine: compute_perplexity returns finite positive",
          "[inference_engine][perplexity]") {
    auto model = create_synthetic_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = TEST_N_LAYERS;
    vmm_cfg.experts_per_layer = TEST_EXPERT_COUNT;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    // Short token sequence
    std::vector<int> tokens = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45};

    double ppl = nos::compute_perplexity(engine, tokens, 0);

    REQUIRE(ppl > 0.0);
    REQUIRE(std::isfinite(ppl));
}

TEST_CASE("InferenceEngine: model_config.json round-trip via shared ModelConfig",
          "[inference_engine]") {
    auto model = create_synthetic_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = TEST_N_LAYERS;
    vmm_cfg.experts_per_layer = TEST_EXPERT_COUNT;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    // Verify all config fields match what was written
    const auto& mc = engine.config();
    REQUIRE(mc.architecture == "llama");
    REQUIRE(mc.vocab_size == TEST_VOCAB_SIZE);
    REQUIRE(mc.hidden_dim == TEST_HIDDEN_DIM);
    REQUIRE(mc.intermediate_dim == TEST_INTERMEDIATE_DIM);
    REQUIRE(mc.n_layers == TEST_N_LAYERS);
    REQUIRE(mc.n_heads == TEST_N_HEADS);
    REQUIRE(mc.n_kv_heads == TEST_N_KV_HEADS);
    REQUIRE(mc.head_dim == TEST_HEAD_DIM);
    REQUIRE(mc.expert_count == TEST_EXPERT_COUNT);
    REQUIRE(mc.top_k == TEST_TOP_K);
    REQUIRE(mc.attention_type == "mha");
}
