/// @file main.cpp
/// @brief NeuralOS CLI: convert, run, train, merge-lora, perplexity, and serve subcommands.

#include "converter/conversion_pipeline.h"
#include "converter/model_config.h"
#include "dashboard/dashboard.h"
#include "engine/benchmark.h"
#include "engine/inference_engine.h"
#include "engine/perplexity.h"
#include "engine/sampling.h"
#include "server/http_server.h"
#include "server/request_scheduler.h"
#include "tokenizer/tokenizer.h"
#include "training/lora.h"
#include "training/trainer.h"
#include "format/expert_format.h"
#include "kernel/packing.h"
#include "converter/quantizer.h"
#include "vmm/memory_budget.h"
#include "vmm/vmm.h"

#include <atomic>
#include <cctype>
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

#include <httplib.h>
#include <nlohmann/json.hpp>

static void print_usage() {
    std::fprintf(stderr,
        "Usage: neuralos <subcommand> [options]\n\n"
        "Subcommands:\n"
        "  convert     Convert a model to NeuralOS .nxp format\n"
        "  run         Generate text from a converted model\n"
        "  bench       Full benchmark suite with paper Table 1 reproduction\n"
        "  train       Train or fine-tune a model\n"
        "  merge-lora  Merge LoRA adapter into base model\n"
        "  serve       Start OpenAI-compatible HTTP server\n"
        "  dashboard   Live TUI dashboard for a running server\n"
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

// ── Train subcommand ────────────────────────────────────────────────────────

static void print_train_usage() {
    std::fprintf(stderr,
        "Usage: neuralos train [options]\n\n"
        "Options:\n"
        "  --model PATH              Converted model directory (required)\n"
        "  --data PATH               JSONL training data file (required)\n"
        "  --output PATH             Output directory (required)\n"
        "  --method {full,lora}      Training method (default: lora)\n"
        "  --memory SIZE             Memory budget (default: 8G)\n"
        "  --epochs INT              Number of epochs (default: 1)\n"
        "  --batch-size INT          Mini-batch size (default: 4)\n"
        "  --lr FLOAT                Learning rate (default: 1e-4)\n"
        "  --lora-rank INT           LoRA rank (default: 16)\n"
        "  --lora-alpha FLOAT        LoRA alpha (default: 16.0)\n"
        "  --steps-per-block INT     BAdam steps per block (default: 100)\n"
        "  --help                    Show this help\n");
}

static int cmd_train(int argc, char** argv) {
    nos::TrainConfig cfg;
    std::string memory_str = "8G";

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_train_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            cfg.data_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            cfg.method = argv[++i];
        } else if (std::strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            memory_str = argv[++i];
        } else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            cfg.max_epochs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            cfg.batch_size = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            cfg.badam_config.lr = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--lora-rank") == 0 && i + 1 < argc) {
            cfg.lora_config.rank = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--lora-alpha") == 0 && i + 1 < argc) {
            cfg.lora_config.alpha = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--steps-per-block") == 0 && i + 1 < argc) {
            cfg.badam_config.steps_per_block = std::atoi(argv[++i]);
        }
    }

    if (cfg.model_dir.empty() || cfg.data_path.empty() || cfg.output_dir.empty()) {
        std::fprintf(stderr, "ERROR: --model, --data, and --output are required\n");
        print_train_usage();
        return 1;
    }

    cfg.memory_budget = nos::parse_memory_string(memory_str);

    nos::Trainer trainer;
    if (!trainer.train(cfg)) {
        std::fprintf(stderr, "ERROR: Training failed\n");
        return 1;
    }

    return 0;
}

// ── Merge-LoRA subcommand ───────────────────────────────────────────────────

static void print_merge_usage() {
    std::fprintf(stderr,
        "Usage: neuralos merge-lora [options]\n\n"
        "Options:\n"
        "  --model PATH              Base model directory (required)\n"
        "  --adapter PATH            Adapter directory from training (required)\n"
        "  --output PATH             Output merged model directory (required)\n"
        "  --help                    Show this help\n");
}

static int cmd_merge(int argc, char** argv) {
    std::string model_dir, adapter_dir, output_dir;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_merge_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--adapter") == 0 && i + 1 < argc) {
            adapter_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    if (model_dir.empty() || adapter_dir.empty() || output_dir.empty()) {
        std::fprintf(stderr, "ERROR: --model, --adapter, and --output are required\n");
        print_merge_usage();
        return 1;
    }

    std::fprintf(stderr, "Merging LoRA adapters...\n");
    std::fprintf(stderr, "  Base model: %s\n", model_dir.c_str());
    std::fprintf(stderr, "  Adapters:   %s\n", adapter_dir.c_str());
    std::fprintf(stderr, "  Output:     %s\n", output_dir.c_str());

    // Read training metadata to find adapter list
    std::string meta_path = adapter_dir + "/training_metadata.json";
    std::ifstream meta_ifs(meta_path);
    if (!meta_ifs.is_open()) {
        std::fprintf(stderr, "ERROR: Cannot read %s\n", meta_path.c_str());
        return 1;
    }

    nlohmann::json meta;
    try {
        meta = nlohmann::json::parse(meta_ifs);
    } catch (const nlohmann::json::exception& e) {
        std::fprintf(stderr, "ERROR: Invalid training metadata: %s\n", e.what());
        return 1;
    }

    if (!meta.contains("adapter_names")) {
        std::fprintf(stderr, "ERROR: No adapter_names in training metadata\n");
        return 1;
    }

    auto adapter_names = meta["adapter_names"].get<std::vector<std::string>>();
    std::fprintf(stderr, "  Found %zu adapters\n", adapter_names.size());

    // Create output directory and copy base model files
    std::filesystem::create_directories(output_dir);

    // Copy model_config.json and tokenizer files
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        const auto& src = entry.path();
        if (src.extension() == ".json" || src.extension() == ".model") {
            auto dst = std::filesystem::path(output_dir) / src.filename();
            std::filesystem::copy_file(src, dst,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    // Load all adapters
    std::vector<nos::LoRAAdapter> adapters;
    std::vector<std::string> loaded_names;
    for (const auto& name : adapter_names) {
        std::string adir = adapter_dir + "/" + name;
        nos::LoRAAdapter adapter;
        if (!adapter.load(adir)) {
            std::fprintf(stderr, "  WARNING: Failed to load adapter %s, skipping\n",
                         name.c_str());
            continue;
        }
        std::fprintf(stderr, "  Loaded adapter: %s (rank=%zu, dims=%zux%zu)\n",
                     name.c_str(), adapter.rank(),
                     adapter.output_dim(), adapter.input_dim());
        adapters.push_back(std::move(adapter));
        loaded_names.push_back(name);
    }

    // Find base .nxp file
    std::string nxp_path;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".nxp") {
            nxp_path = entry.path().string();
            break;
        }
    }

    // Read model config for dimensions
    std::string cfg_path = model_dir + "/model_config.json";
    std::ifstream cfg_ifs(cfg_path);
    nlohmann::json model_cfg;
    uint32_t merge_hidden = 0, merge_intermediate = 0;
    uint32_t merge_n_layers = 0, merge_experts_per_layer = 0;
    if (cfg_ifs.is_open()) {
        try {
            model_cfg = nlohmann::json::parse(cfg_ifs);
            merge_hidden = model_cfg.value("hidden_dim", 0u);
            merge_intermediate = model_cfg.value("intermediate_dim", 0u);
            merge_n_layers = model_cfg.value("n_layers", 0u);
            merge_experts_per_layer = model_cfg.value("expert_count", 0u);
        } catch (...) {}
    }

    size_t merged_count = 0;

    // Merge adapters into .nxp expert weights and write output
    if (!nxp_path.empty()) {
        nos::NxpReader reader;
        if (!reader.open(nxp_path)) {
            std::fprintf(stderr, "ERROR: Failed to open base .nxp: %s\n", nxp_path.c_str());
            return 1;
        }

        std::string out_nxp = output_dir + "/model.nxp";
        nos::NxpWriter writer;
        nos::NxpFileHeader out_hdr = reader.header();
        if (!writer.open(out_nxp, out_hdr)) {
            std::fprintf(stderr, "ERROR: Failed to create output .nxp: %s\n", out_nxp.c_str());
            return 1;
        }

        // Process each expert entry
        for (uint32_t layer = 0; layer < merge_n_layers; ++layer) {
            uint32_t n_exp = (merge_experts_per_layer > 0) ? merge_experts_per_layer : 1;
            for (uint32_t exp_id = 0; exp_id < n_exp; ++exp_id) {
                const nos::NxpExpertEntry* entry = reader.find_expert(layer, exp_id);
                if (entry == nullptr) continue;

                // Read packed weights and scales
                std::vector<uint8_t> packed_buf(static_cast<size_t>(entry->size));
                int rd = reader.read_expert(*entry, packed_buf.data(), packed_buf.size());
                std::vector<uint16_t> scale_buf(static_cast<size_t>(entry->num_channels));
                int sd = reader.read_scales(*entry, scale_buf.data(),
                                            static_cast<size_t>(entry->scale_size));

                if (rd <= 0 || sd <= 0) {
                    // Copy unchanged on read failure
                    writer.write_expert(layer, exp_id,
                                        packed_buf.data(), packed_buf.size(),
                                        scale_buf.data(), entry->num_channels);
                    continue;
                }

                // Check if any adapter targets this expert
                bool has_match = false;
                for (size_t a = 0; a < loaded_names.size(); ++a) {
                    // Adapter names look like "layer0_q_proj" or similar
                    // MoE expert adapters would contain the layer number
                    std::string prefix = "layer" + std::to_string(layer);
                    if (loaded_names[a].find(prefix) != 0) continue;

                    // Dequantize expert weights to FP32
                    int rows = static_cast<int>(merge_hidden);
                    int cols = static_cast<int>(merge_intermediate);
                    if (rows == 0 || cols == 0) {
                        rows = static_cast<int>(entry->num_channels);
                        cols = static_cast<int>(entry->size * 5 /
                               static_cast<size_t>(rows));
                    }
                    int pkd_cols = (cols + 4) / 5;
                    size_t total_params = static_cast<size_t>(rows) *
                                          static_cast<size_t>(cols);
                    std::vector<float> weights_fp32(total_params);
                    std::vector<int8_t> trits(static_cast<size_t>(cols));

                    for (int r = 0; r < rows; ++r) {
                        const uint8_t* row_p = packed_buf.data()
                            + static_cast<size_t>(r) * static_cast<size_t>(pkd_cols);
                        nos::unpack_row(row_p, cols, trits.data());
                        float scale = nos::fp16_to_fp32(
                            scale_buf[static_cast<size_t>(r)]);
                        for (int c = 0; c < cols; ++c) {
                            weights_fp32[static_cast<size_t>(r) *
                                         static_cast<size_t>(cols) +
                                         static_cast<size_t>(c)] =
                                static_cast<float>(trits[static_cast<size_t>(c)]) * scale;
                        }
                    }

                    // Apply merge_into
                    adapters[a].merge_into(weights_fp32.data(),
                                           static_cast<size_t>(rows),
                                           static_cast<size_t>(cols));

                    // Re-quantize: FP32 -> FP16 -> ternary
                    std::vector<uint16_t> fp16_buf(total_params);
                    for (size_t i = 0; i < total_params; ++i) {
                        fp16_buf[i] = nos::fp32_to_fp16(weights_fp32[i]);
                    }
                    nos::QuantizedWeights qw = nos::ternary_quantize(
                        fp16_buf.data(), rows, cols);

                    // Write re-quantized weights
                    writer.write_expert(layer, exp_id,
                                        qw.packed.data(), qw.packed.size(),
                                        qw.scales.data(),
                                        static_cast<uint32_t>(qw.scales.size()));
                    has_match = true;
                    ++merged_count;
                    std::fprintf(stderr, "  Merged adapter %s into layer %u expert %u\n",
                                 loaded_names[a].c_str(), layer, exp_id);
                    break;
                }

                if (!has_match) {
                    // No adapter for this expert -- copy unchanged
                    writer.write_expert(layer, exp_id,
                                        packed_buf.data(), packed_buf.size(),
                                        scale_buf.data(), entry->num_channels);
                }
            }
        }

        writer.finalize();
        std::fprintf(stderr, "Output .nxp written: %s\n", out_nxp.c_str());
    } else {
        std::fprintf(stderr, "WARNING: No .nxp file found in base model, skipping weight merge\n");
    }

    std::fprintf(stderr, "Merge complete: %zu adapters applied\n", merged_count);
    std::fprintf(stderr, "Output model: %s\n", output_dir.c_str());
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

// ── Dashboard subcommand ────────────────────────────────────────────────────

static void print_dashboard_usage() {
    std::fprintf(stderr,
        "Usage: neuralos dashboard [options]\n\n"
        "Options:\n"
        "  --host STRING             Server host (default: 127.0.0.1)\n"
        "  --port INT                Server port (default: 8080)\n"
        "  --refresh-ms INT          Refresh interval in ms (default: 500)\n"
        "  --help                    Show this help\n");
}

static int cmd_dashboard(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    int refresh_ms = 500;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_dashboard_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--refresh-ms") == 0 && i + 1 < argc) {
            refresh_ms = std::atoi(argv[++i]);
            if (refresh_ms < 100) refresh_ms = 100;
        }
    }

    // Connect to server health endpoint to discover shm_name.
    // Use httplib client for the health check.
    std::string base_url = host + ":" + std::to_string(port);
    httplib::Client client(host, port);
    client.set_connection_timeout(2);  // 2 second timeout
    client.set_read_timeout(2);

    auto res = client.Get("/health");
    if (!res || res->status != 200) {
        std::fprintf(stderr,
            "Cannot connect to server at %s:%d\n"
            "Make sure 'neuralos serve' is running.\n",
            host.c_str(), port);
        return 1;
    }

    // Parse health response for shm_name
    std::string shm_name;
    try {
        auto j = nlohmann::json::parse(res->body);
        if (j.contains("shm_name")) {
            shm_name = j["shm_name"].get<std::string>();
        }
    } catch (...) {
        std::fprintf(stderr, "Invalid health response from server\n");
        return 1;
    }

    if (shm_name.empty()) {
        std::fprintf(stderr, "Server did not report shared metrics name\n");
        return 1;
    }

    std::fprintf(stderr, "Connected to server at %s:%d (shm: %s)\n",
                 host.c_str(), port, shm_name.c_str());

    nos::Dashboard dashboard;
    nos::Dashboard::Config cfg;
    cfg.shm_name = shm_name;
    cfg.refresh_ms = refresh_ms;

    if (!dashboard.start(cfg)) {
        return 1;
    }

    dashboard.run();
    dashboard.stop();
    return 0;
}

// ── Bench subcommand ────────────────────────────────────────────────────────

static const char* kDefaultBenchPrompt =
    "Explain the concept of quantum entanglement in simple terms. "
    "Then write a Python function to calculate fibonacci numbers. "
    "Finally, describe the process of photosynthesis in plants.";

static void print_bench_usage() {
    std::fprintf(stderr,
        "Usage: neuralos bench [options]\n\n"
        "Options:\n"
        "  --model PATH              Converted model directory (required)\n"
        "  --memory SIZE             Memory budget (default: 16G)\n"
        "  --output-dir PATH         Output directory (default: benchmark_results)\n"
        "  --threads N               Thread count (default: 0=auto)\n"
        "  --tokens INT              Tokens per run (default: 512)\n"
        "  --prompt TEXT              Benchmark prompt (default: built-in multi-domain prompt)\n"
        "  --prefetch                Enable oracle prefetcher\n"
        "  --prefetch-k INT          Max lookahead depth (default: 10)\n"
        "  --standalone              LaTeX standalone mode\n"
        "  --concurrent INT          Multi-sequence batch count (default: 4)\n"
        "  --compare DIR             Merge with previous results from DIR\n"
        "  --skip-multi-seq          Skip multi-sequence batch test\n"
        "  --help                    Show this help\n");
}

static int cmd_bench(int argc, char** argv) {
    std::string model_dir, memory_str = "16G";
    std::string output_dir = "benchmark_results";
    std::string prompt = kDefaultBenchPrompt;
    std::string compare_dir;
    int threads = 0;
    int tokens = 512;
    bool prefetch_enabled = false;
    int  prefetch_max_k = 10;
    bool standalone_latex = false;
    int  concurrent = 4;
    bool skip_multi_seq = false;

    for (int i = 0; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            print_bench_usage();
            return 0;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            memory_str = argv[++i];
        } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--prefetch") == 0) {
            prefetch_enabled = true;
        } else if (std::strcmp(argv[i], "--prefetch-k") == 0 && i + 1 < argc) {
            prefetch_max_k = std::atoi(argv[++i]);
            if (prefetch_max_k < 1)  prefetch_max_k = 1;
            if (prefetch_max_k > 10) prefetch_max_k = 10;
        } else if (std::strcmp(argv[i], "--standalone") == 0) {
            standalone_latex = true;
        } else if (std::strcmp(argv[i], "--concurrent") == 0 && i + 1 < argc) {
            concurrent = std::atoi(argv[++i]);
            if (concurrent < 1) concurrent = 1;
        } else if (std::strcmp(argv[i], "--compare") == 0 && i + 1 < argc) {
            compare_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--skip-multi-seq") == 0) {
            skip_multi_seq = true;
        }
    }

    if (model_dir.empty()) {
        std::fprintf(stderr, "ERROR: --model is required\n");
        print_bench_usage();
        return 1;
    }

    size_t memory_budget = nos::parse_memory_string(memory_str);
    size_t memory_budget_mb = memory_budget / (1024ULL * 1024ULL);

    // Load tokenizer
    nos::Tokenizer tokenizer;
    std::string tok_path = find_tokenizer(model_dir);
    if (!tok_path.empty()) {
        tokenizer.load(tok_path);
    }

    // Encode prompt helper
    auto encode_prompt = [&](const std::string& p) -> std::vector<int> {
        std::vector<int> ids;
        if (tokenizer.is_loaded()) {
            ids = tokenizer.encode(p);
            if (tokenizer.bos_id() >= 0) {
                ids.insert(ids.begin(), tokenizer.bos_id());
            }
        } else {
            for (unsigned char c : p) {
                ids.push_back(static_cast<int>(c));
            }
        }
        return ids;
    };

    std::vector<nos::BenchRunResult> results;

    // Helper to extract model size from model_dir name (e.g., "70b", "7B")
    auto extract_model_size = [](const std::string& dir) -> std::string {
        std::filesystem::path p(dir);
        std::string name = p.filename().string();
        for (size_t i = 0; i < name.size(); i++) {
            if (std::isdigit(static_cast<unsigned char>(name[i]))) {
                size_t j = i;
                while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) {
                    ++j;
                }
                if (j < name.size() && (name[j] == 'b' || name[j] == 'B')) {
                    return name.substr(i, j - i + 1);
                }
            }
        }
        return "?B";
    };

    std::string model_size = extract_model_size(model_dir);

    std::fprintf(stderr, "=== NeuralOS Benchmark Suite ===\n");
    std::fprintf(stderr, "Model: %s\n", model_dir.c_str());
    std::fprintf(stderr, "Memory: %zuMB | Threads: %d\n\n", memory_budget_mb, threads);

    // ── Run 1: Baseline (no prefetch) ──────────────────────────────────

    {
        std::fprintf(stderr, "Run 1 (baseline): generating %d tokens...\n", tokens);
        nos::VmmFullConfig vmm_cfg;
        vmm_cfg.nxp_path = model_dir + "/model.nxp";
        vmm_cfg.user_budget_bytes = memory_budget;
        auto vmm = nos::Vmm::create(vmm_cfg);
        if (!vmm) {
            std::fprintf(stderr, "ERROR: Failed to create VMM for baseline run\n");
            return 1;
        }

        nos::InferenceEngine engine;
        engine.set_sticky_config(-1.0f, 128);
        engine.set_prefetch_config(false, 0);
        if (!engine.load(model_dir, vmm.get(), threads)) {
            std::fprintf(stderr, "ERROR: Failed to load model\n");
            return 1;
        }

        auto prompt_ids = encode_prompt(prompt);
        auto t_start = std::chrono::high_resolution_clock::now();

        // Prefill
        const float* logits = nullptr;
        for (size_t i = 0; i < prompt_ids.size(); i++) {
            logits = engine.forward_step(prompt_ids[i], static_cast<int>(i));
        }
        auto t_first = std::chrono::high_resolution_clock::now();
        double ttft = std::chrono::duration<double, std::milli>(t_first - t_start).count();

        // Generate
        std::mt19937 rng(42);
        nos::SamplingParams sp;
        sp.temperature = 0.0f;  // greedy for reproducible benchmarks
        std::vector<int> context = prompt_ids;
        int pos = static_cast<int>(prompt_ids.size());
        int gen = 0;

        for (int t = 0; t < tokens && logits != nullptr; t++) {
            std::vector<float> lc(logits, logits + engine.vocab_size());
            int next = nos::sample(lc.data(), engine.vocab_size(), sp, context, rng);
            if (tokenizer.is_loaded() && next == tokenizer.eos_id()) break;
            context.push_back(next);
            logits = engine.forward_step(next, pos++);
            ++gen;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tps = (elapsed > 0.0)
            ? static_cast<double>(gen) / (elapsed / 1000.0) : 0.0;

        auto routing = engine.routing_metrics();
        auto vmm_stats = vmm->stats();
        double chr = (vmm_stats.total_pins > 0)
            ? static_cast<double>(vmm_stats.cache_hits) / static_cast<double>(vmm_stats.total_pins)
            : 0.0;

        nos::BenchRunResult r;
        r.model_name         = model_dir;
        r.model_size         = model_size;
        r.tokens_generated   = gen;
        r.total_time_ms      = elapsed;
        r.ttft_ms            = ttft;
        r.tok_per_sec        = tps;
        r.cache_hit_rate     = chr;
        r.switch_rate        = static_cast<double>(routing.switch_rate);
        r.avg_sticky_window  = static_cast<double>(routing.avg_window_length);
        r.prefetch_mode      = "none";
        r.effective_k        = 0;
        r.prefetch_rwp       = 0.0;
        r.waste_ratio        = 0.0;
        r.memory_budget_mb   = memory_budget_mb;
        r.num_threads        = threads;
        results.push_back(std::move(r));

        std::fprintf(stderr, "  %.1f tok/s | TTFT: %.0fms | Cache: %.1f%%\n",
                     tps, ttft, chr * 100.0);
    }

    // ── Run 2: With oracle prefetch (if --prefetch) ────────────────────

    if (prefetch_enabled) {
        std::fprintf(stderr, "Run 2 (oracle prefetch): generating %d tokens...\n", tokens);
        nos::VmmFullConfig vmm_cfg;
        vmm_cfg.nxp_path = model_dir + "/model.nxp";
        vmm_cfg.user_budget_bytes = memory_budget;
        auto vmm = nos::Vmm::create(vmm_cfg);
        if (!vmm) {
            std::fprintf(stderr, "ERROR: Failed to create VMM for oracle run\n");
            return 1;
        }

        nos::InferenceEngine engine;
        engine.set_sticky_config(-1.0f, 128);
        engine.set_prefetch_config(true, prefetch_max_k);
        if (!engine.load(model_dir, vmm.get(), threads)) {
            std::fprintf(stderr, "ERROR: Failed to load model\n");
            return 1;
        }

        auto prompt_ids = encode_prompt(prompt);
        auto t_start = std::chrono::high_resolution_clock::now();

        const float* logits = nullptr;
        for (size_t i = 0; i < prompt_ids.size(); i++) {
            logits = engine.forward_step(prompt_ids[i], static_cast<int>(i));
        }
        auto t_first = std::chrono::high_resolution_clock::now();
        double ttft = std::chrono::duration<double, std::milli>(t_first - t_start).count();

        std::mt19937 rng(42);
        nos::SamplingParams sp;
        sp.temperature = 0.0f;
        std::vector<int> context = prompt_ids;
        int pos = static_cast<int>(prompt_ids.size());
        int gen = 0;

        for (int t = 0; t < tokens && logits != nullptr; t++) {
            std::vector<float> lc(logits, logits + engine.vocab_size());
            int next = nos::sample(lc.data(), engine.vocab_size(), sp, context, rng);
            if (tokenizer.is_loaded() && next == tokenizer.eos_id()) break;
            context.push_back(next);
            logits = engine.forward_step(next, pos++);
            ++gen;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double tps = (elapsed > 0.0)
            ? static_cast<double>(gen) / (elapsed / 1000.0) : 0.0;

        auto routing = engine.routing_metrics();
        auto vmm_stats = vmm->stats();
        double chr = (vmm_stats.total_pins > 0)
            ? static_cast<double>(vmm_stats.cache_hits) / static_cast<double>(vmm_stats.total_pins)
            : 0.0;

        auto pstats = engine.prefetch_stats();

        nos::BenchRunResult r;
        r.model_name         = model_dir;
        r.model_size         = model_size;
        r.tokens_generated   = gen;
        r.total_time_ms      = elapsed;
        r.ttft_ms            = ttft;
        r.tok_per_sec        = tps;
        r.cache_hit_rate     = chr;
        r.switch_rate        = static_cast<double>(routing.switch_rate);
        r.avg_sticky_window  = static_cast<double>(routing.avg_window_length);
        r.prefetch_mode      = pstats.mode;
        r.effective_k        = pstats.effective_k;
        r.prefetch_rwp       = pstats.rwp_oracle;
        r.waste_ratio        = (pstats.speculative_hits + pstats.speculative_misses > 0)
            ? static_cast<double>(pstats.speculative_misses)
              / static_cast<double>(pstats.speculative_hits + pstats.speculative_misses)
            : 0.0;
        r.memory_budget_mb   = memory_budget_mb;
        r.num_threads        = threads;
        results.push_back(std::move(r));

        std::fprintf(stderr, "  %.1f tok/s | TTFT: %.0fms | Cache: %.1f%% | RWP: %.1f%%\n",
                     tps, ttft, chr * 100.0, pstats.rwp_oracle * 100.0);
    }

    // ── Run 3: Multi-sequence batch test ───────────────────────────────

    if (!skip_multi_seq) {
        std::fprintf(stderr, "Run 3 (multi-seq x%d): measuring concurrent throughput...\n",
                     concurrent);

        nos_config_t base_cfg;
        std::memset(&base_cfg, 0, sizeof(base_cfg));
        base_cfg.struct_size = sizeof(nos_config_t);
        base_cfg.temperature = 0.0f;
        base_cfg.top_k = 1;
        base_cfg.top_p = 1.0f;
        base_cfg.repetition_penalty = 1.0f;
        base_cfg.min_p = 0.0f;
        base_cfg.sticky_lambda = -1.0f;
        base_cfg.sticky_window = 128;
        base_cfg.num_threads = threads;
        base_cfg.memory_budget = memory_budget;
        base_cfg.prefetch_enabled = prefetch_enabled ? 1 : 0;
        base_cfg.prefetch_max_k = prefetch_max_k;

        nos::RequestScheduler scheduler(
            static_cast<size_t>(concurrent), model_dir, base_cfg);

        if (!scheduler.is_ready()) {
            std::fprintf(stderr, "  WARNING: RequestScheduler failed to initialize, skipping\n");
        } else {
            int tokens_per_seq = tokens / concurrent;
            if (tokens_per_seq < 1) tokens_per_seq = 1;

            std::atomic<int> total_tokens_generated{0};
            auto t_start = std::chrono::high_resolution_clock::now();

            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(concurrent));

            for (int s = 0; s < concurrent; s++) {
                workers.emplace_back([&, s]() {
                    (void)s;  // unused but kept for clarity
                    auto guard = scheduler.acquire_slot_guard();
                    if (!guard) return;

                    auto* slot = guard.get();
                    if (slot == nullptr || slot->ctx == nullptr) return;

                    auto* ctx = slot->ctx;
                    int generated = 0;
                    int input_token = 1;  // start token
                    for (int t = 0; t < tokens_per_seq; t++) {
                        int out_token = 0;
                        int rc = nos_step_token(ctx, input_token, &out_token);
                        if (rc != NOS_OK) break;
                        input_token = out_token;
                        ++generated;
                    }
                    total_tokens_generated.fetch_add(generated,
                                                     std::memory_order_relaxed);
                });
            }

            for (auto& w : workers) {
                w.join();
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(
                t_end - t_start).count();
            int total_gen = total_tokens_generated.load(std::memory_order_relaxed);
            double agg_tps = (wall_ms > 0.0)
                ? static_cast<double>(total_gen) / (wall_ms / 1000.0) : 0.0;

            nos::BenchRunResult r;
            r.model_name           = model_dir;
            r.model_size           = model_size;
            r.tokens_generated     = total_gen;
            r.total_time_ms        = wall_ms;
            r.tok_per_sec          = agg_tps;
            r.prefetch_mode        = prefetch_enabled ? "oracle" : "none";
            r.memory_budget_mb     = memory_budget_mb;
            r.num_threads          = threads;
            r.concurrent_sequences = concurrent;
            r.multi_seq_tok_per_sec = agg_tps;
            results.push_back(std::move(r));

            std::fprintf(stderr, "  %.1f tok/s aggregate (%d concurrent)\n",
                         agg_tps, concurrent);
        }
    }

    // ── Output generation ──────────────────────────────────────────────

    nos::BenchmarkReporter reporter({{output_dir}, standalone_latex});

    std::filesystem::create_directories(output_dir);
    reporter.write_paper_table(output_dir + "/table1.tex", results);
    reporter.write_paper_csv(output_dir + "/benchmark.csv", results);
    reporter.write_paper_json(output_dir + "/benchmark.json", results);

    // ── Comparison mode ────────────────────────────────────────────────

    if (!compare_dir.empty()) {
        std::string prev_json_path = compare_dir + "/benchmark.json";
        std::ifstream prev_ifs(prev_json_path);
        if (prev_ifs.is_open()) {
            try {
                auto prev_arr = nlohmann::json::parse(prev_ifs);
                std::vector<nos::BenchRunResult> merged = results;
                for (const auto& j : prev_arr) {
                    nos::BenchRunResult r;
                    r.model_name       = j.value("model_name", "");
                    r.model_size       = j.value("model_size", "");
                    r.tokens_generated = j.value("tokens_generated", 0);
                    r.total_time_ms    = j.value("total_time_ms", 0.0);
                    r.ttft_ms          = j.value("ttft_ms", 0.0);
                    r.tok_per_sec      = j.value("tok_per_sec", 0.0);
                    r.cache_hit_rate   = j.value("cache_hit_rate", 0.0);
                    r.switch_rate      = j.value("switch_rate", 0.0);
                    r.avg_sticky_window = j.value("avg_sticky_window", 0.0);
                    r.prefetch_rwp     = j.value("prefetch_rwp", 0.0);
                    r.prefetch_mode    = j.value("prefetch_mode", "none");
                    r.effective_k      = j.value("effective_k", 0);
                    r.waste_ratio      = j.value("waste_ratio", 0.0);
                    r.memory_budget_mb = j.value("memory_budget_mb", static_cast<size_t>(0));
                    r.num_threads      = j.value("num_threads", 0);
                    r.concurrent_sequences = j.value("concurrent_sequences", 0);
                    r.multi_seq_tok_per_sec = j.value("multi_seq_tok_per_sec", 0.0);
                    merged.push_back(std::move(r));
                }
                reporter.write_comparison_table(
                    output_dir + "/comparison.tex", merged);
                std::fprintf(stderr, "  Comparison table: %s/comparison.tex\n",
                             output_dir.c_str());
            } catch (const nlohmann::json::exception& e) {
                std::fprintf(stderr, "  WARNING: Failed to parse %s: %s\n",
                             prev_json_path.c_str(), e.what());
            }
        } else {
            std::fprintf(stderr, "  WARNING: Cannot open %s for comparison\n",
                         prev_json_path.c_str());
        }
    }

    // ── Summary ────────────────────────────────────────────────────────

    std::fprintf(stderr, "\n=== NeuralOS Benchmark Results ===\n");
    std::fprintf(stderr, "Model: %s\n", model_dir.c_str());
    std::fprintf(stderr, "Memory: %zuMB | Threads: %d\n\n", memory_budget_mb, threads);

    size_t run_idx = 0;
    if (run_idx < results.size()) {
        const auto& r = results[run_idx];
        std::fprintf(stderr, "Run 1 (baseline):    %.1f tok/s | TTFT: %.0fms | Cache: %.1f%%\n",
                     r.tok_per_sec, r.ttft_ms, r.cache_hit_rate * 100.0);
        ++run_idx;
    }
    if (prefetch_enabled && run_idx < results.size()) {
        const auto& r = results[run_idx];
        std::fprintf(stderr, "Run 2 (oracle):      %.1f tok/s | TTFT: %.0fms | Cache: %.1f%% | RWP: %.1f%%\n",
                     r.tok_per_sec, r.ttft_ms, r.cache_hit_rate * 100.0,
                     r.prefetch_rwp * 100.0);
        ++run_idx;
    }
    if (!skip_multi_seq && run_idx < results.size()) {
        const auto& r = results[run_idx];
        std::fprintf(stderr, "Run 3 (multi-seq):   %.1f tok/s aggregate (%d concurrent)\n",
                     r.multi_seq_tok_per_sec, r.concurrent_sequences);
    }

    std::fprintf(stderr, "\nOutput: %s/\n", output_dir.c_str());

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
    } else if (std::strcmp(cmd, "bench") == 0) {
        return cmd_bench(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "train") == 0) {
        return cmd_train(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "merge-lora") == 0) {
        return cmd_merge(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "serve") == 0) {
        return cmd_serve(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "dashboard") == 0) {
        return cmd_dashboard(argc - 2, argv + 2);
    } else if (std::strcmp(cmd, "perplexity") == 0) {
        return cmd_perplexity(argc - 2, argv + 2);
    } else {
        std::fprintf(stderr, "Unknown subcommand: %s\n", cmd);
        print_usage();
        return 1;
    }
}
