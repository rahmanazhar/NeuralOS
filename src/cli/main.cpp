/// @file main.cpp
/// @brief NeuralOS CLI: convert, run, perplexity, and serve subcommands.

#include "converter/conversion_pipeline.h"
#include "converter/model_config.h"
#include "engine/benchmark.h"
#include "engine/inference_engine.h"
#include "engine/perplexity.h"
#include "engine/sampling.h"
#include "server/http_server.h"
#include "tokenizer/tokenizer.h"
#include "vmm/memory_budget.h"
#include "vmm/vmm.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

static void print_usage() {
    std::fprintf(stderr,
        "Usage: neuralos <subcommand> [options]\n\n"
        "Subcommands:\n"
        "  convert     Convert a model to NeuralOS .nxp format\n"
        "  run         Generate text from a converted model\n"
        "  serve       Start OpenAI-compatible HTTP server\n"
        "  perplexity  Evaluate model perplexity on a text file\n\n"
        "Use 'neuralos <subcommand> --help' for subcommand-specific options.\n");
}

static void print_convert_usage() {
    std::fprintf(stderr,
        "Usage: neuralos convert [options]\n\n"
        "Options:\n"
        "  --input PATH              Input model directory (required)\n"
        "  --output PATH             Output directory for .nxp (required)\n"
        "  --experts N               Target expert size in MB (default: 100)\n"
        "  --calibration N           Calibration samples (default: 1024)\n"
        "  --calibration-data PATH   Calibration text file (e.g. WikiText-2)\n"
        "  --resume                  Resume from checkpoint\n"
        "  --skip-perplexity-gate    Skip perplexity validation\n"
        "  --help                    Show this help\n");
}

static void print_run_usage() {
    std::fprintf(stderr,
        "Usage: neuralos run [options]\n\n"
        "Options:\n"
        "  --model PATH              Converted model directory (required)\n"
        "  --prompt TEXT              Input prompt (required)\n"
        "  --memory SIZE             Memory budget (default: 8G)\n"
        "  --temperature FLOAT       Sampling temperature (default: 1.0, 0=greedy)\n"
        "  --top-k INT               Top-k filtering (default: 40, 0=disabled)\n"
        "  --top-p FLOAT             Top-p nucleus sampling (default: 0.95)\n"
        "  --repetition-penalty F    Repetition penalty (default: 1.1)\n"
        "  --min-p FLOAT             Min-p filtering (default: 0.05)\n"
        "  --max-tokens INT          Max tokens to generate (default: 256)\n"
        "  --seed INT                Random seed (default: 0=random)\n"
        "  --json                    Output JSON instead of streaming text\n"
        "  --threads N               Thread count for expert-parallel dispatch (0=auto)\n"
        "  --sticky-lambda FLOAT     Override adaptive lambda (-1=auto, default)\n"
        "  --sticky-window INT       Max stickiness window in tokens (default: 128)\n"
        "  --bench                   Benchmark mode: write CSV+JSON+LaTeX to --output-dir\n"
        "  --output-dir PATH         Benchmark output directory (default: benchmark_results)\n"
        "  --standalone              LaTeX standalone mode (wraps with preamble)\n"
        "  --trace-routing           Write per-token routing trace to output-dir\n"
        "  --prefetch               Enable oracle speculative prefetcher (default: off)\n"
        "  --prefetch-k INT         Max lookahead depth (default: 10, range 1-10)\n"
        "  --help                    Show this help\n");
}

static void print_perplexity_usage() {
    std::fprintf(stderr,
        "Usage: neuralos perplexity [options]\n\n"
        "Options:\n"
        "  --model PATH              Converted model directory (required)\n"
        "  --data PATH               Text file for evaluation (required)\n"
        "  --memory SIZE             Memory budget (default: 8G)\n"
        "  --context-length INT      Context length (default: model max)\n"
        "  --help                    Show this help\n");
}

static std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static std::string find_tokenizer(const std::string& dir) {
    // Try common tokenizer file names
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

// ── Convert subcommand ──────────────────────────────────────────────────────

static int cmd_convert(int argc, char** argv) {
    std::string input_path, output_dir, calibration_data_path;
    int expert_size_mb = 100;
    int calibration_samples = 1024;
    bool resume = false;
    bool skip_ppl_gate = false;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_convert_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            expert_size_mb = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--calibration") == 0 && i + 1 < argc) {
            calibration_samples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--calibration-data") == 0 && i + 1 < argc) {
            calibration_data_path = argv[++i];
        } else if (std::strcmp(argv[i], "--resume") == 0) {
            resume = true;
        } else if (std::strcmp(argv[i], "--skip-perplexity-gate") == 0) {
            skip_ppl_gate = true;
        }
    }

    if (input_path.empty() || output_dir.empty()) {
        std::fprintf(stderr, "ERROR: --input and --output are required\n");
        print_convert_usage();
        return 1;
    }

    // Run conversion pipeline
    nos::ConversionConfig cfg;
    cfg.input_path = input_path;
    cfg.output_dir = output_dir;
    cfg.target_expert_size_mb = expert_size_mb;
    cfg.calibration_samples = calibration_samples;
    cfg.top_k = 2;
    cfg.resume = resume;
    cfg.calibration_data_path = calibration_data_path;

    nos::ConversionPipeline pipeline;
    if (!pipeline.run(cfg)) {
        std::fprintf(stderr, "ERROR: Conversion failed\n");
        return 1;
    }
    std::fprintf(stderr, "Conversion complete.\n");

    // Perplexity budget gate (PIPE-12)
    if (!skip_ppl_gate && !calibration_data_path.empty()) {
        std::fprintf(stderr, "Running perplexity budget gate...\n");

        nos::VmmFullConfig vmm_cfg;
        vmm_cfg.nxp_path = output_dir + "/model.nxp";
        vmm_cfg.user_budget_bytes = nos::parse_memory_string("16G");
        auto vmm = nos::Vmm::create(vmm_cfg);
        if (!vmm) {
            std::fprintf(stderr, "ERROR: Failed to create VMM for perplexity gate\n");
            return 1;
        }

        nos::InferenceEngine engine;
        if (!engine.load(output_dir, vmm.get())) {
            std::fprintf(stderr, "ERROR: Failed to load converted model\n");
            return 1;
        }

        nos::Tokenizer tokenizer;
        std::string tok_path = find_tokenizer(input_path);
        if (tok_path.empty() || !tokenizer.load(tok_path)) {
            std::fprintf(stderr, "WARNING: No tokenizer found, skipping perplexity gate\n");
            return 0;
        }

        auto tokens = tokenizer.encode(read_file(calibration_data_path));
        if (tokens.empty()) {
            std::fprintf(stderr, "WARNING: Empty calibration data, skipping perplexity gate\n");
            return 0;
        }

        double ppl = nos::compute_perplexity(engine, tokens);
        std::fprintf(stderr, "Converted model perplexity: %.2f\n", ppl);

        constexpr double FP16_BASELINE_PPL = 5.47;
        constexpr double MAX_ALLOWED_PPL = FP16_BASELINE_PPL * 1.05;
        if (ppl > MAX_ALLOWED_PPL) {
            std::fprintf(stderr, "ERROR: Perplexity %.2f exceeds budget (max %.2f). "
                        "Conversion rejected.\n", ppl, MAX_ALLOWED_PPL);
            return 1;
        }
        std::fprintf(stderr, "Perplexity gate PASSED (%.2f <= %.2f)\n", ppl, MAX_ALLOWED_PPL);
    }

    return 0;
}

// ── Run subcommand ──────────────────────────────────────────────────────────

static int cmd_run(int argc, char** argv) {
    std::string model_dir, prompt, memory_str = "8G";
    std::string output_dir = "benchmark_results";
    nos::SamplingParams params;
    int max_tokens = 256;
    int threads = 0;
    float sticky_lambda = -1.0f;
    int sticky_window = 128;
    bool json_mode = false;
    bool bench_mode = false;
    bool standalone_latex = false;
    bool trace_routing = false;
    bool prefetch_enabled = false;
    int  prefetch_max_k   = 10;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_run_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            memory_str = argv[++i];
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            params.temperature = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            params.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            params.top_p = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--repetition-penalty") == 0 && i + 1 < argc) {
            params.repetition_penalty = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            params.min_p = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            params.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json_mode = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sticky-lambda") == 0 && i + 1 < argc) {
            sticky_lambda = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--sticky-window") == 0 && i + 1 < argc) {
            sticky_window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench_mode = true;
        } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--standalone") == 0) {
            standalone_latex = true;
        } else if (std::strcmp(argv[i], "--trace-routing") == 0) {
            trace_routing = true;
        } else if (std::strcmp(argv[i], "--prefetch") == 0) {
            prefetch_enabled = true;
        } else if (std::strcmp(argv[i], "--prefetch-k") == 0 && i + 1 < argc) {
            prefetch_max_k = std::atoi(argv[++i]);
            if (prefetch_max_k < 1)  prefetch_max_k = 1;
            if (prefetch_max_k > 10) prefetch_max_k = 10;
        }
    }

    if (model_dir.empty() || prompt.empty()) {
        std::fprintf(stderr, "ERROR: --model and --prompt are required\n");
        print_run_usage();
        return 1;
    }

    // Load tokenizer
    nos::Tokenizer tokenizer;
    std::string tok_path = find_tokenizer(model_dir);
    if (!tok_path.empty()) {
        tokenizer.load(tok_path);
    }

    // Create VMM
    nos::VmmFullConfig vmm_cfg;
    vmm_cfg.nxp_path = model_dir + "/model.nxp";
    vmm_cfg.user_budget_bytes = nos::parse_memory_string(memory_str);
    auto vmm = nos::Vmm::create(vmm_cfg);
    if (!vmm) {
        std::fprintf(stderr, "ERROR: Failed to create VMM\n");
        return 1;
    }

    // Load engine with thread count
    nos::InferenceEngine engine;
    engine.set_sticky_config(sticky_lambda, sticky_window);
    engine.set_prefetch_config(prefetch_enabled, prefetch_max_k);
    if (!engine.load(model_dir, vmm.get(), threads)) {
        std::fprintf(stderr, "ERROR: Failed to load model\n");
        return 1;
    }

    // Set up RNG
    std::mt19937 rng;
    if (params.seed != 0) {
        rng.seed(static_cast<std::mt19937::result_type>(params.seed));
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    // Open routing trace file if requested
    std::ofstream trace_file;
    if (trace_routing) {
        std::filesystem::create_directories(output_dir);
        trace_file.open(output_dir + "/routing_trace.jsonl");
    }

    // Encode prompt
    std::vector<int> prompt_ids;
    if (tokenizer.is_loaded()) {
        prompt_ids = tokenizer.encode(prompt);
        if (tokenizer.bos_id() >= 0) {
            prompt_ids.insert(prompt_ids.begin(), tokenizer.bos_id());
        }
    } else {
        // Fallback: byte encoding
        for (unsigned char c : prompt) {
            prompt_ids.push_back(static_cast<int>(c));
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill: process prompt tokens
    std::vector<int> context = prompt_ids;
    const float* logits = nullptr;
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        logits = engine.forward_step(prompt_ids[i], static_cast<int>(i));
    }

    auto t_first = std::chrono::high_resolution_clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();

    // Generate
    int pos = static_cast<int>(prompt_ids.size());
    std::string generated_text;
    int generated_count = 0;
    int eos_id = tokenizer.is_loaded() ? tokenizer.eos_id() : -1;

    for (int t = 0; t < max_tokens && logits != nullptr; t++) {
        // Copy logits for sampling (transforms modify in-place)
        std::vector<float> logits_copy(logits, logits + engine.vocab_size());
        int next_token = nos::sample(logits_copy.data(), engine.vocab_size(),
                                     params, context, rng);

        if (next_token == eos_id) break;

        context.push_back(next_token);
        generated_count++;

        // Decode and stream
        if (tokenizer.is_loaded()) {
            auto text = tokenizer.decode({next_token});
            generated_text += text;
            if (!json_mode) {
                std::fprintf(stdout, "%s", text.c_str());
                std::fflush(stdout);
            }
        } else {
            if (next_token >= 0 && next_token < 128) {
                char c = static_cast<char>(next_token);
                generated_text += c;
                if (!json_mode) {
                    std::fputc(c, stdout);
                    std::fflush(stdout);
                }
            }
        }

        logits = engine.forward_step(next_token, pos++);

        // Write routing trace entry
        if (trace_routing && trace_file.is_open()) {
            const auto& tr = engine.last_routing_trace();
            nlohmann::json tj;
            tj["token_pos"] = tr.token_pos;
            tj["layer_id"] = tr.layer_id;
            tj["lambda"] = tr.lambda;
            tj["io_pressure"] = tr.io_pressure;
            tj["ppl_delta"] = tr.ppl_delta;
            tj["switched"] = tr.switched;
            tj["reason"] = tr.reason;
            tj["candidate_experts"] = tr.candidate_experts;
            trace_file << tj.dump() << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double tok_per_sec = (elapsed_ms > 0)
        ? static_cast<double>(generated_count) / (elapsed_ms / 1000.0)
        : 0.0;

    if (json_mode) {
        // JSON output
        std::fprintf(stdout, "{\"text\": \"");
        for (char c : generated_text) {
            if (c == '"') std::fprintf(stdout, "\\\"");
            else if (c == '\\') std::fprintf(stdout, "\\\\");
            else if (c == '\n') std::fprintf(stdout, "\\n");
            else if (c == '\r') std::fprintf(stdout, "\\r");
            else if (c == '\t') std::fprintf(stdout, "\\t");
            else std::fputc(c, stdout);
        }
        std::fprintf(stdout, "\", \"tokens\": %d, \"time_ms\": %.1f, \"tok_per_sec\": %.1f}\n",
                     generated_count, elapsed_ms, tok_per_sec);
    } else {
        std::fprintf(stdout, "\n");
    }

    // Always print summary stats to stderr
    auto routing = engine.routing_metrics();
    nos::VmmStats vmm_stats = vmm->stats();
    double cache_hit_rate = (vmm_stats.total_pins > 0)
        ? static_cast<double>(vmm_stats.cache_hits)
          / static_cast<double>(vmm_stats.total_pins)
        : 0.0;

    std::fprintf(stderr,
        "\n--- Summary ---\n"
        "  tok/s: %.2f | TTFT: %.1f ms | tokens: %d\n"
        "  cache hit rate: %.2f%% | switch rate: %.4f | avg window: %.1f\n",
        tok_per_sec, ttft_ms, generated_count,
        cache_hit_rate * 100.0,
        static_cast<double>(routing.switch_rate),
        static_cast<double>(routing.avg_window_length));

    // Benchmark mode: write CSV + JSON + LaTeX
    if (bench_mode) {
        nos::BenchmarkReporter reporter({{output_dir}, standalone_latex});
        reporter.set_run_info(model_dir, generated_count, elapsed_ms, ttft_ms);
        if (prefetch_enabled) {
            reporter.write(engine.metrics(), vmm_stats, routing,
                           engine.prefetch_stats());
        } else {
            reporter.write(engine.metrics(), vmm_stats, routing);
        }
        std::fprintf(stderr, "  Benchmark output written to: %s/\n", output_dir.c_str());
    }

    // Close trace file
    if (trace_file.is_open()) {
        trace_file.close();
        std::fprintf(stderr, "  Routing trace written to: %s/routing_trace.jsonl\n",
                     output_dir.c_str());
    }

    return 0;
}

// ── Perplexity subcommand ───────────────────────────────────────────────────

static int cmd_perplexity(int argc, char** argv) {
    std::string model_dir, data_path, memory_str = "8G";
    int context_length = 0;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_perplexity_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (std::strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            memory_str = argv[++i];
        } else if (std::strcmp(argv[i], "--context-length") == 0 && i + 1 < argc) {
            context_length = std::atoi(argv[++i]);
        }
    }

    if (model_dir.empty() || data_path.empty()) {
        std::fprintf(stderr, "ERROR: --model and --data are required\n");
        print_perplexity_usage();
        return 1;
    }

    // Load model
    nos::VmmFullConfig vmm_cfg;
    vmm_cfg.nxp_path = model_dir + "/model.nxp";
    vmm_cfg.user_budget_bytes = nos::parse_memory_string(memory_str);
    auto vmm = nos::Vmm::create(vmm_cfg);
    if (!vmm) {
        std::fprintf(stderr, "ERROR: Failed to create VMM\n");
        return 1;
    }

    nos::InferenceEngine engine;
    if (!engine.load(model_dir, vmm.get())) {
        std::fprintf(stderr, "ERROR: Failed to load model\n");
        return 1;
    }

    // Load tokenizer
    nos::Tokenizer tokenizer;
    std::string tok_path = find_tokenizer(model_dir);
    if (!tok_path.empty()) {
        tokenizer.load(tok_path);
    }

    // Read and tokenize data
    std::string text = read_file(data_path);
    if (text.empty()) {
        std::fprintf(stderr, "ERROR: Could not read %s\n", data_path.c_str());
        return 1;
    }

    std::vector<int> tokens;
    if (tokenizer.is_loaded()) {
        tokens = tokenizer.encode(text);
    } else {
        for (unsigned char c : text) {
            tokens.push_back(static_cast<int>(c));
        }
    }

    int ctx_len = context_length > 0
        ? context_length
        : static_cast<int>(engine.config().max_seq_len);

    double ppl = nos::compute_perplexity(engine, tokens, ctx_len);
    int chunks = static_cast<int>(tokens.size()) / ctx_len;

    std::fprintf(stdout, "Perplexity: %.4f (on %zu tokens, %d chunks)\n",
                 ppl, tokens.size(), chunks);

    constexpr double FP16_BASELINE_PPL = 5.47;
    if (ppl > FP16_BASELINE_PPL * 1.05) {
        std::fprintf(stderr, "WARNING: Perplexity %.2f exceeds 5%% of FP16 baseline (%.2f)\n",
                     ppl, FP16_BASELINE_PPL);
    }

    return 0;
}

// ── Serve subcommand ────────────────────────────────────────────────────────

static void print_serve_usage() {
    std::fprintf(stderr,
        "Usage: neuralos serve [options]\n\n"
        "Options:\n"
        "  --model PATH              Converted model directory (required)\n"
        "  --port INT                Server port (default: 8080)\n"
        "  --host STRING             Bind address (default: 127.0.0.1)\n"
        "  --memory SIZE             Memory budget (default: 8G)\n"
        "  --temperature FLOAT       Sampling temperature (default: 1.0)\n"
        "  --top-k INT               Top-k filtering (default: 40)\n"
        "  --top-p FLOAT             Top-p nucleus sampling (default: 0.95)\n"
        "  --threads N               Thread count (0=auto)\n"
        "  --sticky-lambda FLOAT     Override adaptive lambda (-1=auto)\n"
        "  --sticky-window INT       Max stickiness window (default: 128)\n"
        "  --prefetch                Enable oracle prefetcher\n"
        "  --prefetch-k INT          Max lookahead depth (default: 10)\n"
        "  --help                    Show this help\n");
}

static volatile std::sig_atomic_t g_running = 1;

static void signal_handler(int /*sig*/) {
    g_running = 0;
}

static int cmd_serve(int argc, char** argv) {
    std::string model_dir, memory_str = "8G", host = "127.0.0.1";
    int port = 8080;
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.95f;
    int threads = 0;
    float sticky_lambda = -1.0f;
    int sticky_window = 128;
    bool prefetch_enabled = false;
    int  prefetch_max_k = 10;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_serve_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (std::strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            memory_str = argv[++i];
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sticky-lambda") == 0 && i + 1 < argc) {
            sticky_lambda = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--sticky-window") == 0 && i + 1 < argc) {
            sticky_window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prefetch") == 0) {
            prefetch_enabled = true;
        } else if (std::strcmp(argv[i], "--prefetch-k") == 0 && i + 1 < argc) {
            prefetch_max_k = std::atoi(argv[++i]);
            if (prefetch_max_k < 1)  prefetch_max_k = 1;
            if (prefetch_max_k > 10) prefetch_max_k = 10;
        }
    }

    if (model_dir.empty()) {
        std::fprintf(stderr, "ERROR: --model is required\n");
        print_serve_usage();
        return 1;
    }

    nos::HttpServer::Config srv_cfg;
    srv_cfg.host = host;
    srv_cfg.port = port;
    srv_cfg.model_path = model_dir;

    std::memset(&srv_cfg.inference_config, 0, sizeof(nos_config_t));
    srv_cfg.inference_config.struct_size = sizeof(nos_config_t);
    srv_cfg.inference_config.temperature = temperature;
    srv_cfg.inference_config.top_k = top_k;
    srv_cfg.inference_config.top_p = top_p;
    srv_cfg.inference_config.repetition_penalty = 1.1f;
    srv_cfg.inference_config.min_p = 0.05f;
    srv_cfg.inference_config.sticky_lambda = sticky_lambda;
    srv_cfg.inference_config.sticky_window = sticky_window;
    srv_cfg.inference_config.num_threads = threads;
    srv_cfg.inference_config.memory_budget = nos::parse_memory_string(memory_str);
    srv_cfg.inference_config.prefetch_enabled = prefetch_enabled ? 1 : 0;
    srv_cfg.inference_config.prefetch_max_k = prefetch_max_k;

    nos::HttpServer server;
    if (!server.start(srv_cfg)) {
        std::fprintf(stderr, "ERROR: Failed to start HTTP server\n");
        return 1;
    }

    std::fprintf(stderr, "Listening on http://%s:%d\n", host.c_str(), port);
    std::fprintf(stderr, "Press Ctrl+C to stop.\n");

    // Wait for SIGINT
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    while (g_running != 0 && server.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::fprintf(stderr, "\nShutting down...\n");
    server.stop();
    return 0;
}

// ── Main entry ──────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char* cmd = argv[1];

    if (std::strcmp(cmd, "--help") == 0 || std::strcmp(cmd, "-h") == 0) {
        print_usage();
        return 0;
    } else if (std::strcmp(cmd, "convert") == 0) {
        return cmd_convert(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "run") == 0) {
        return cmd_run(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "serve") == 0) {
        return cmd_serve(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "perplexity") == 0) {
        return cmd_perplexity(argc - 2, argv + 2);
    } else {
        std::fprintf(stderr, "Unknown subcommand: %s\n", cmd);
        print_usage();
        return 1;
    }
}
