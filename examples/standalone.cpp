/// @file standalone.cpp
/// @brief Example program using the NeuralOS C API (libneuralos).
///
/// Compile (against installed library):
///   g++ -std=c++20 standalone.cpp -lneuralos -I/usr/local/include -L/usr/local/lib -o standalone
///
/// Usage:
///   ./standalone /path/to/converted-model "Your prompt here"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "api/libneuralos.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model_dir> <prompt>\n", argv[0]);
        std::fprintf(stderr, "\nNeuralOS C API version: %s\n", nos_version());
        return 1;
    }

    const char* model_dir = argv[1];
    const char* prompt = argv[2];

    // Configure inference
    nos_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size = sizeof(nos_config_t);
    cfg.temperature = 0.7f;
    cfg.top_k = 40;
    cfg.top_p = 0.95f;
    cfg.repetition_penalty = 1.1f;
    cfg.min_p = 0.05f;
    cfg.sticky_lambda = -1.0f;  // auto
    cfg.sticky_window = 128;
    cfg.num_threads = 0;  // auto

    // Create context
    nos_ctx_t* ctx = nos_create(model_dir, cfg);
    if (ctx == nullptr) {
        std::fprintf(stderr, "ERROR: Failed to create context: %s\n",
                     nos_last_error());
        return 1;
    }

    // Generate text
    char output[4096];
    int rc = nos_generate(ctx, prompt, output, sizeof(output));
    if (rc != NOS_OK) {
        std::fprintf(stderr, "ERROR: nos_generate failed (rc=%d): %s\n",
                     rc, nos_last_error());
        nos_destroy(ctx);
        return 1;
    }

    std::printf("%s\n", output);

    // Print metrics
    const char* metrics = nos_get_metrics(ctx);
    if (metrics != nullptr) {
        std::fprintf(stderr, "\nMetrics: %s\n", metrics);
    }

    nos_destroy(ctx);
    return 0;
}
