/// @file lora.cpp
/// @brief LoRA adapter implementation.

#include "training/lora.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include <nlohmann/json.hpp>

namespace nos {

LoRAAdapter::LoRAAdapter(LoRAConfig config, size_t input_dim, size_t output_dim)
    : config_(std::move(config))
    , input_dim_(input_dim)
    , output_dim_(output_dim)
{
    const size_t r = config_.rank;

    // A: rank x input_dim, Kaiming He init: N(0, sqrt(2/input_dim))
    A_.resize(r * input_dim);
    {
        std::mt19937 rng(42);
        const float stddev = std::sqrt(2.0f / static_cast<float>(input_dim));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (size_t i = 0; i < r * input_dim; ++i) {
            A_[i] = dist(rng);
        }
    }

    // B: output_dim x rank, zero init
    B_.assign(output_dim * r, 0.0f);
}

void LoRAAdapter::forward(const float* x, float* delta,
                          size_t batch_size) const {
    const size_t r = config_.rank;
    const float scale = config_.alpha / static_cast<float>(r);

    for (size_t b = 0; b < batch_size; ++b) {
        const float* x_b = x + b * input_dim_;
        float* delta_b = delta + b * output_dim_;

        // Step 1: tmp = A * x (rank vector)
        std::vector<float> tmp(r, 0.0f);
        for (size_t i = 0; i < r; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < input_dim_; ++j) {
                sum += A_[i * input_dim_ + j] * x_b[j];
            }
            tmp[i] = sum;
        }

        // Step 2: delta = B * tmp (output_dim vector), scaled by alpha/rank
        for (size_t i = 0; i < output_dim_; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < r; ++j) {
                sum += B_[i * r + j] * tmp[j];
            }
            delta_b[i] = scale * sum;
        }
    }
}

void LoRAAdapter::backward(const float* x, const float* grad_output,
                           float* grad_A, float* grad_B) const {
    const size_t r = config_.rank;
    const float scale = config_.alpha / static_cast<float>(r);

    // Compute A * x for this input
    std::vector<float> Ax(r, 0.0f);
    for (size_t i = 0; i < r; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_dim_; ++j) {
            sum += A_[i * input_dim_ + j] * x[j];
        }
        Ax[i] = sum;
    }

    // grad_B = scale * grad_output * (A*x)^T  (output_dim x rank)
    for (size_t i = 0; i < output_dim_; ++i) {
        for (size_t j = 0; j < r; ++j) {
            grad_B[i * r + j] = scale * grad_output[i] * Ax[j];
        }
    }

    // Compute B^T * grad_output (rank vector)
    std::vector<float> Bt_grad(r, 0.0f);
    for (size_t j = 0; j < r; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < output_dim_; ++i) {
            sum += B_[i * r + j] * grad_output[i];
        }
        Bt_grad[j] = sum;
    }

    // grad_A = scale * B^T * grad_output * x^T  (rank x input_dim)
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < input_dim_; ++j) {
            grad_A[i * input_dim_ + j] = scale * Bt_grad[i] * x[j];
        }
    }
}

void LoRAAdapter::update(const float* grad_A, const float* grad_B, float lr) {
    const size_t r = config_.rank;

    // SGD update: A -= lr * grad_A
    for (size_t i = 0; i < r * input_dim_; ++i) {
        A_[i] -= lr * grad_A[i];
    }

    // SGD update: B -= lr * grad_B
    for (size_t i = 0; i < output_dim_ * r; ++i) {
        B_[i] -= lr * grad_B[i];
    }
}

void LoRAAdapter::merge_into(float* W, size_t rows, size_t cols) const {
    // W += (alpha/rank) * B * A
    // B is output_dim x rank, A is rank x input_dim
    // Result: output_dim x input_dim (rows x cols)
    const size_t r = config_.rank;
    const float scale = config_.alpha / static_cast<float>(r);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < r; ++k) {
                sum += B_[i * r + k] * A_[k * cols + j];
            }
            W[i * cols + j] += scale * sum;
        }
    }
}

bool LoRAAdapter::save(const std::string& dir) const {
    std::filesystem::create_directories(dir);

    // Save metadata
    nlohmann::json meta;
    meta["rank"] = config_.rank;
    meta["alpha"] = config_.alpha;
    meta["input_dim"] = input_dim_;
    meta["output_dim"] = output_dim_;
    meta["target_layers"] = config_.target_layers;

    {
        std::ofstream ofs(dir + "/lora_adapter.json");
        if (!ofs.is_open()) {
            std::fprintf(stderr, "LoRA: failed to write metadata to %s\n", dir.c_str());
            return false;
        }
        ofs << meta.dump(2);
    }

    // Save A matrix (raw float32)
    {
        std::ofstream ofs(dir + "/lora_A.bin", std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(A_.data()),
                  static_cast<std::streamsize>(A_.size() * sizeof(float)));
    }

    // Save B matrix (raw float32)
    {
        std::ofstream ofs(dir + "/lora_B.bin", std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(B_.data()),
                  static_cast<std::streamsize>(B_.size() * sizeof(float)));
    }

    return true;
}

bool LoRAAdapter::load(const std::string& dir) {
    // Load metadata
    std::ifstream meta_ifs(dir + "/lora_adapter.json");
    if (!meta_ifs.is_open()) {
        std::fprintf(stderr, "LoRA: failed to read metadata from %s\n", dir.c_str());
        return false;
    }

    nlohmann::json meta;
    try {
        meta = nlohmann::json::parse(meta_ifs);
    } catch (const nlohmann::json::exception& e) {
        std::fprintf(stderr, "LoRA: invalid metadata JSON: %s\n", e.what());
        return false;
    }

    const size_t loaded_rank = meta.at("rank").get<size_t>();
    const float loaded_alpha = meta.at("alpha").get<float>();
    const size_t loaded_input = meta.at("input_dim").get<size_t>();
    const size_t loaded_output = meta.at("output_dim").get<size_t>();

    // Validate dimensions before overwriting (per 05-02 decision pattern)
    if (input_dim_ > 0 && (loaded_input != input_dim_ || loaded_output != output_dim_)) {
        std::fprintf(stderr, "LoRA: dimension mismatch: expected %zux%zu, got %zux%zu\n",
                     input_dim_, output_dim_, loaded_input, loaded_output);
        return false;
    }

    config_.rank = loaded_rank;
    config_.alpha = loaded_alpha;
    input_dim_ = loaded_input;
    output_dim_ = loaded_output;
    if (meta.contains("target_layers")) {
        config_.target_layers = meta["target_layers"].get<std::vector<std::string>>();
    }

    // Load A matrix
    const size_t a_size = loaded_rank * loaded_input;
    A_.resize(a_size);
    {
        std::ifstream ifs(dir + "/lora_A.bin", std::ios::binary);
        if (!ifs.is_open()) return false;
        ifs.read(reinterpret_cast<char*>(A_.data()),
                 static_cast<std::streamsize>(a_size * sizeof(float)));
        if (!ifs) return false;
    }

    // Load B matrix
    const size_t b_size = loaded_output * loaded_rank;
    B_.resize(b_size);
    {
        std::ifstream ifs(dir + "/lora_B.bin", std::ios::binary);
        if (!ifs.is_open()) return false;
        ifs.read(reinterpret_cast<char*>(B_.data()),
                 static_cast<std::streamsize>(b_size * sizeof(float)));
        if (!ifs) return false;
    }

    return true;
}

size_t LoRAAdapter::rank() const {
    return config_.rank;
}

float LoRAAdapter::alpha() const {
    return config_.alpha;
}

size_t LoRAAdapter::param_count() const {
    return config_.rank * (input_dim_ + output_dim_);
}

size_t LoRAAdapter::input_dim() const {
    return input_dim_;
}

size_t LoRAAdapter::output_dim() const {
    return output_dim_;
}

float* LoRAAdapter::A_data() {
    return A_.data();
}

const float* LoRAAdapter::A_data() const {
    return A_.data();
}

float* LoRAAdapter::B_data() {
    return B_.data();
}

const float* LoRAAdapter::B_data() const {
    return B_.data();
}

}  // namespace nos
