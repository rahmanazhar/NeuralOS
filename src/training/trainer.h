#pragma once

/// @file trainer.h
/// @brief Training orchestrator coordinating BAdam/GaLore/LoRA.
///
/// Provides two training modes:
///   - "full": BAdam block-wise training with GaLore for shared layers
///   - "lora": LoRA fine-tuning with frozen base weights
///
/// NOTE: The forward/backward pass is SIMPLIFIED for this research prototype.
/// Full autograd is not implemented. Instead, per-expert gradient computation
/// uses analytical gradients for linear layers (grad_W = x^T * grad_output).
/// This demonstrates the training algorithms on real expert weights.

#include "training/badam.h"
#include "training/galore.h"
#include "training/lora.h"

#include <cstddef>
#include <string>

namespace nos {

/// Training configuration.
struct TrainConfig {
    std::string model_dir;              ///< Input model directory
    std::string data_path;              ///< JSONL training data path
    std::string output_dir;             ///< Output directory
    std::string method = "full";        ///< "full" or "lora"
    BAdamConfig badam_config;           ///< BAdam hyperparameters
    GaLoreConfig galore_config;         ///< GaLore hyperparameters
    LoRAConfig lora_config;             ///< LoRA hyperparameters
    size_t memory_budget = 0;           ///< Memory budget in bytes (0 = auto)
    int max_epochs = 1;                 ///< Number of training epochs
    size_t batch_size = 4;              ///< Mini-batch size
    size_t max_seq_len = 512;           ///< Maximum sequence length
};

/// Training orchestrator.
///
/// Coordinates BAdam (full training) or LoRA (fine-tuning) modes.
/// Full training: sequential block-wise training with GaLore for shared layers.
/// LoRA: frozen base weights with low-rank adapters on attention projections.
class Trainer {
public:
    Trainer() = default;

    /// Main entry point. Returns true on success.
    bool train(const TrainConfig& config);

private:
    /// Full training mode: BAdam block-wise + GaLore for shared layers.
    bool train_full(const TrainConfig& config);

    /// LoRA fine-tuning mode.
    bool train_lora(const TrainConfig& config);
};

}  // namespace nos
