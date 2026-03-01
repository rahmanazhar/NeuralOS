#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "converter/model_config.h"
#include "engine/inference_engine.h"
#include "engine/perplexity.h"
#include "engine/sampling.h"
#include "format/expert_format.h"
#include "kernel/packing.h"
#include "vmm/vmm.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

// Synthetic model parameters
static constexpr uint32_t E2E_VOCAB = 256;
static constexpr uint32_t E2E_HIDDEN = 64;
static constexpr uint32_t E2E_INTER = 128;
static constexpr uint32_t E2E_LAYERS = 2;
static constexpr uint32_t E2E_HEADS = 4;
static constexpr uint32_t E2E_KV_HEADS = 4;
static constexpr uint32_t E2E_HEAD_DIM = 16;  // hidden / heads
static constexpr uint32_t E2E_EXPERTS = 2;
static constexpr uint32_t E2E_TOP_K = 2;
static constexpr uint32_t E2E_SEQ = 128;

namespace {

struct SyntheticModel {
    std::string dir;
    ~SyntheticModel() { std::filesystem::remove_all(dir); }
};

SyntheticModel create_e2e_model(uint32_t n_kv_heads = E2E_KV_HEADS) {
    namespace fs = std::filesystem;

    auto sz = [](uint32_t v) -> size_t { return static_cast<size_t>(v); };

    std::string tmp_base = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp")
                         + "/nos_e2e_test";
    fs::create_directories(tmp_base);

    SyntheticModel model;
    model.dir = tmp_base;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    nos::ModelConfig mc;
    mc.architecture = "llama";
    mc.vocab_size = E2E_VOCAB;
    mc.hidden_dim = E2E_HIDDEN;
    mc.intermediate_dim = E2E_INTER;
    mc.n_layers = E2E_LAYERS;
    mc.n_heads = E2E_HEADS;
    mc.n_kv_heads = n_kv_heads;
    mc.head_dim = E2E_HEAD_DIM;
    mc.max_seq_len = E2E_SEQ;
    mc.rope_theta = 10000.0f;
    mc.norm_eps = 1e-5f;
    mc.expert_count = E2E_EXPERTS;
    mc.top_k = E2E_TOP_K;
    mc.max_expert_size = 0;
    mc.attention_type = (n_kv_heads == 1) ? "mqa"
                      : (n_kv_heads < E2E_HEADS) ? "gqa" : "mha";

    // Create .nxp
    nos::NxpFileHeader header{};
    header.magic = nos::NXP_MAGIC;
    header.version = nos::NXP_VERSION;
    header.num_layers = E2E_LAYERS;
    header.experts_per_layer = E2E_EXPERTS;
    header.hidden_dim = E2E_HIDDEN;
    header.intermediate_dim = E2E_INTER;
    header.packing_mode = 0;
    header.scale_dtype = 0;
    header.total_experts = sz(E2E_LAYERS) * sz(E2E_EXPERTS);
    std::memset(header.reserved, 0, sizeof(header.reserved));

    std::string nxp_path = model.dir + "/model.nxp";
    nos::NxpWriter writer;
    REQUIRE(writer.open(nxp_path, header));

    uint64_t max_expert_bytes = 0;

    for (uint32_t L = 0; L < E2E_LAYERS; L++) {
        // Expert ternary weights
        for (uint32_t E = 0; E < E2E_EXPERTS; E++) {
            int expert_rows = static_cast<int>(E2E_INTER / E2E_EXPERTS);
            int gu_rows = expert_rows * 2;
            int packed_cols_gu = (static_cast<int>(E2E_HIDDEN) + 4) / 5;
            int down_rows = static_cast<int>(E2E_HIDDEN);
            int packed_cols_down = (expert_rows + 4) / 5;

            // Gate + up ternary weights
            std::vector<int8_t> gu_trits(sz(gu_rows) * sz(E2E_HIDDEN));
            std::uniform_int_distribution<int> trit_dist(-1, 1);
            for (auto& t : gu_trits) t = static_cast<int8_t>(trit_dist(rng));

            std::vector<uint8_t> gu_packed(sz(gu_rows) * sz(packed_cols_gu));
            for (int r = 0; r < gu_rows; r++) {
                nos::pack_row(gu_trits.data() + sz(r) * sz(E2E_HIDDEN),
                             static_cast<int>(E2E_HIDDEN),
                             gu_packed.data() + sz(r) * sz(packed_cols_gu));
            }

            // Down projection ternary weights [hidden_dim x expert_rows]
            std::vector<int8_t> down_trits(sz(down_rows) * sz(expert_rows));
            for (auto& t : down_trits) t = static_cast<int8_t>(trit_dist(rng));

            std::vector<uint8_t> down_packed(sz(down_rows) * sz(packed_cols_down));
            for (int r = 0; r < down_rows; r++) {
                nos::pack_row(down_trits.data() + sz(r) * sz(expert_rows),
                             expert_rows,
                             down_packed.data() + sz(r) * sz(packed_cols_down));
            }

            // Concatenate packed data and scales
            std::vector<uint8_t> all_packed;
            all_packed.insert(all_packed.end(), gu_packed.begin(), gu_packed.end());
            all_packed.insert(all_packed.end(), down_packed.begin(), down_packed.end());

            uint32_t total_channels = static_cast<uint32_t>(gu_rows + down_rows);
            std::vector<uint16_t> all_scales(sz(total_channels));
            for (auto& s : all_scales) s = nos::fp32_to_fp16(std::abs(dist(rng)) + 0.01f);

            auto entry = writer.write_expert(L, E, all_packed.data(), all_packed.size(),
                                              all_scales.data(), total_channels);
            if (entry.size > max_expert_bytes) max_expert_bytes = entry.size;
        }

        // INT8 attention weights
        uint32_t attn_ids[] = {0xFFFFFFF0u, 0xFFFFFFF1u, 0xFFFFFFF2u, 0xFFFFFFF3u};
        int q_rows = static_cast<int>(E2E_HEADS * E2E_HEAD_DIM);
        int k_rows = static_cast<int>(n_kv_heads * E2E_HEAD_DIM);
        int v_rows = k_rows;
        int o_rows = static_cast<int>(E2E_HIDDEN);
        int attn_rows[] = {q_rows, k_rows, v_rows, o_rows};
        int attn_cols[] = {static_cast<int>(E2E_HIDDEN),
                          static_cast<int>(E2E_HIDDEN),
                          static_cast<int>(E2E_HIDDEN),
                          q_rows};

        for (int a = 0; a < 4; a++) {
            std::vector<int8_t> data(sz(attn_rows[a]) * sz(attn_cols[a]));
            std::uniform_int_distribution<int> byte_dist(-127, 127);
            for (auto& b : data) b = static_cast<int8_t>(byte_dist(rng));

            std::vector<uint16_t> scales(sz(attn_rows[a]));
            for (auto& s : scales) s = nos::fp32_to_fp16(std::abs(dist(rng)) + 0.01f);

            writer.write_expert(L, attn_ids[a],
                               reinterpret_cast<const uint8_t*>(data.data()),
                               data.size(), scales.data(),
                               static_cast<uint32_t>(attn_rows[a]));
        }

        // Router weights
        std::vector<uint16_t> router_fp16(sz(E2E_EXPERTS) * sz(E2E_HIDDEN));
        for (auto& w : router_fp16) w = nos::fp32_to_fp16(dist(rng));
        writer.write_expert(L, 0xFFFFFFFFu,
                           reinterpret_cast<const uint8_t*>(router_fp16.data()),
                           router_fp16.size() * sizeof(uint16_t), nullptr, 0);
    }

    // Embedding
    std::vector<uint16_t> embedding(sz(E2E_VOCAB) * sz(E2E_HIDDEN));
    for (auto& e : embedding) e = nos::fp32_to_fp16(dist(rng));
    writer.write_expert(UINT32_MAX, 0,
                       reinterpret_cast<const uint8_t*>(embedding.data()),
                       embedding.size() * sizeof(uint16_t), nullptr, 0);

    // Output projection
    std::vector<uint16_t> output_proj(sz(E2E_VOCAB) * sz(E2E_HIDDEN));
    for (auto& o : output_proj) o = nos::fp32_to_fp16(dist(rng));
    writer.write_expert(UINT32_MAX, 1,
                       reinterpret_cast<const uint8_t*>(output_proj.data()),
                       output_proj.size() * sizeof(uint16_t), nullptr, 0);

    // RMSNorm weights
    size_t total_norm = (sz(E2E_LAYERS) * 2 + 1) * sz(E2E_HIDDEN);
    std::vector<uint16_t> norm_fp16(total_norm);
    for (auto& n : norm_fp16) n = nos::fp32_to_fp16(1.0f);
    writer.write_expert(UINT32_MAX, 2,
                       reinterpret_cast<const uint8_t*>(norm_fp16.data()),
                       norm_fp16.size() * sizeof(uint16_t), nullptr, 0);

    REQUIRE(writer.finalize());

    mc.max_expert_size = static_cast<uint32_t>(max_expert_bytes);

    // Write model_config.json
    nlohmann::json j = mc;
    std::ofstream ofs(model.dir + "/model_config.json");
    ofs << j.dump(2);
    ofs.close();

    return model;
}

}  // anonymous namespace

TEST_CASE("E2E: synthetic model generates 10 valid tokens", "[e2e]") {
    auto model = create_e2e_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = E2E_LAYERS;
    vmm_cfg.experts_per_layer = E2E_EXPERTS;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);
    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    // Generate 10 tokens
    std::vector<int> context = {1};  // BOS
    const float* logits = nullptr;

    for (int i = 0; i < 10; i++) {
        logits = engine.forward_step(context.back(), i);
        REQUIRE(logits != nullptr);

        // Verify logits are valid
        float max_logit = *std::max_element(logits, logits + engine.vocab_size());
        float sum_exp = 0.0f;
        bool valid = true;
        for (int v = 0; v < engine.vocab_size(); v++) {
            if (std::isnan(logits[v]) || std::isinf(logits[v])) {
                valid = false;
                break;
            }
            sum_exp += std::exp(logits[v] - max_logit);
        }
        REQUIRE(valid);
        REQUIRE(sum_exp > 0.0f);
        REQUIRE(std::isfinite(sum_exp));

        // Greedy sample
        int next = nos::greedy_sample(logits, engine.vocab_size());
        REQUIRE(next >= 0);
        REQUIRE(next < engine.vocab_size());
        context.push_back(next);
    }

    REQUIRE(context.size() == 11);
}

TEST_CASE("E2E: model_config.json schema round-trip", "[e2e]") {
    auto model = create_e2e_model();

    // Read back model_config.json
    std::ifstream ifs(model.dir + "/model_config.json");
    REQUIRE(ifs.is_open());
    nlohmann::json j = nlohmann::json::parse(ifs);
    nos::ModelConfig mc = j.get<nos::ModelConfig>();

    REQUIRE(mc.vocab_size == E2E_VOCAB);
    REQUIRE(mc.hidden_dim == E2E_HIDDEN);
    REQUIRE(mc.intermediate_dim == E2E_INTER);
    REQUIRE(mc.n_layers == E2E_LAYERS);
    REQUIRE(mc.n_heads == E2E_HEADS);
    REQUIRE(mc.n_kv_heads == E2E_KV_HEADS);
    REQUIRE(mc.head_dim == E2E_HEAD_DIM);
    REQUIRE(mc.expert_count == E2E_EXPERTS);
    REQUIRE(mc.top_k == E2E_TOP_K);
    REQUIRE(mc.max_expert_size > 0);
    REQUIRE(mc.attention_type == "mha");
}

TEST_CASE("E2E: KV cache reset produces identical output", "[e2e]") {
    auto model = create_e2e_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = E2E_LAYERS;
    vmm_cfg.experts_per_layer = E2E_EXPERTS;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);
    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    // Run 1
    const float* logits1 = engine.forward_step(1, 0);
    std::vector<float> saved1(logits1, logits1 + engine.vocab_size());

    // Reset and run again
    engine.reset_kv_cache();
    const float* logits2 = engine.forward_step(1, 0);

    // Should produce identical logits
    for (int v = 0; v < engine.vocab_size(); v++) {
        REQUIRE_THAT(logits2[v], WithinAbs(saved1[static_cast<size_t>(v)], 1e-6f));
    }
}

TEST_CASE("E2E: GQA configuration produces valid output", "[e2e]") {
    // n_kv_heads=2, n_heads=4 -> GQA with group size 2
    auto model = create_e2e_model(2);

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = E2E_LAYERS;
    vmm_cfg.experts_per_layer = E2E_EXPERTS;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);
    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    const float* logits = engine.forward_step(1, 0);
    REQUIRE(logits != nullptr);

    bool valid = true;
    for (int v = 0; v < engine.vocab_size(); v++) {
        if (std::isnan(logits[v]) || std::isinf(logits[v])) {
            valid = false;
            break;
        }
    }
    REQUIRE(valid);
    REQUIRE(engine.config().attention_type == "gqa");
}

TEST_CASE("E2E: MQA configuration produces valid output", "[e2e]") {
    // n_kv_heads=1 -> MQA
    auto model = create_e2e_model(1);

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = E2E_LAYERS;
    vmm_cfg.experts_per_layer = E2E_EXPERTS;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);
    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    const float* logits = engine.forward_step(1, 0);
    REQUIRE(logits != nullptr);

    bool valid = true;
    for (int v = 0; v < engine.vocab_size(); v++) {
        if (std::isnan(logits[v]) || std::isinf(logits[v])) {
            valid = false;
            break;
        }
    }
    REQUIRE(valid);
    REQUIRE(engine.config().attention_type == "mqa");
}

TEST_CASE("E2E: sampling with full parameter suite", "[e2e]") {
    auto model = create_e2e_model();

    nos::VmmConfig vmm_cfg;
    vmm_cfg.expert_cache_bytes = 1024 * 1024;
    vmm_cfg.max_expert_size = 4096;
    vmm_cfg.num_layers = E2E_LAYERS;
    vmm_cfg.experts_per_layer = E2E_EXPERTS;
    vmm_cfg.nxp_path = model.dir + "/model.nxp";

    nos::Vmm vmm(vmm_cfg);
    nos::InferenceEngine engine;
    REQUIRE(engine.load(model.dir, &vmm));

    nos::SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 10;
    params.top_p = 0.9f;
    params.repetition_penalty = 1.2f;
    params.min_p = 0.05f;
    std::mt19937 rng(42);
    std::vector<int> context = {1};

    // Generate 5 tokens with full sampling suite
    for (int i = 0; i < 5; i++) {
        const float* logits = engine.forward_step(context.back(), i);
        REQUIRE(logits != nullptr);

        std::vector<float> logits_copy(logits, logits + engine.vocab_size());
        int next = nos::sample(logits_copy.data(), engine.vocab_size(),
                              params, context, rng);
        REQUIRE(next >= 0);
        REQUIRE(next < engine.vocab_size());
        context.push_back(next);
    }

    REQUIRE(context.size() == 6);
}

// ── 7B Integration Test (env-gated) ─────────────────────────────────────────

TEST_CASE("7B model integration", "[.integration][7b]") {
    const char* model_path = std::getenv("NOS_TEST_7B_MODEL");
    if (!model_path) {
        SKIP("Set NOS_TEST_7B_MODEL to Llama 2 7B SafeTensors directory");
    }

    const char* converted_path = std::getenv("NOS_TEST_7B_CONVERTED");
    std::string conv_dir;
    if (converted_path) {
        conv_dir = converted_path;
    } else {
        SKIP("Set NOS_TEST_7B_CONVERTED to pre-converted model directory");
    }

    nos::VmmFullConfig vmm_cfg;
    vmm_cfg.nxp_path = conv_dir + "/model.nxp";
    vmm_cfg.user_budget_bytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    auto vmm = nos::Vmm::create(vmm_cfg);
    REQUIRE(vmm != nullptr);

    nos::InferenceEngine engine;
    REQUIRE(engine.load(conv_dir, vmm.get()));

    // Generate tokens (greedy)
    std::vector<int> context = {1};  // BOS
    nos::SamplingParams params;
    params.temperature = 0.0f;
    std::mt19937 rng(42);

    for (int i = 0; i < 50; i++) {
        const float* logits = engine.forward_step(context.back(), i);
        REQUIRE(logits != nullptr);

        std::vector<float> logits_copy(logits, logits + engine.vocab_size());
        int next = nos::sample(logits_copy.data(), engine.vocab_size(),
                              params, context, rng);
        context.push_back(next);
    }

    INFO("Generated " << context.size() << " tokens");
    REQUIRE(context.size() == 51);

    // WikiText-2 perplexity validation (PIPE-12 / VALD-01)
    const char* wikitext_path = std::getenv("NOS_TEST_WIKITEXT2");
    if (wikitext_path) {
        std::ifstream wf(wikitext_path);
        std::string wikitext_content((std::istreambuf_iterator<char>(wf)),
                                      std::istreambuf_iterator<char>());

        // Byte-level tokenization for perplexity
        std::vector<int> wikitext_tokens;
        for (unsigned char c : wikitext_content) {
            wikitext_tokens.push_back(static_cast<int>(c));
        }

        engine.reset_kv_cache();
        double perplexity = nos::compute_perplexity(engine, wikitext_tokens);
        INFO("WikiText-2 perplexity: " << perplexity);

        constexpr double FP16_BASELINE_PPL = 5.47;
        constexpr double MAX_ALLOWED_PPL = FP16_BASELINE_PPL * 1.05;
        REQUIRE(perplexity < MAX_ALLOWED_PPL);
    }
}
