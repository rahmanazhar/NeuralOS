/// @file safetensors_reader.cpp
/// @brief SafeTensors format reader implementation.

#include "converter/safetensors_reader.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace nos {

SafeTensorsReader::SafeTensorsReader() = default;

SafeTensorsReader::~SafeTensorsReader() {
    close_all();
}

bool SafeTensorsReader::open(const std::string& path) {
    close_all();
    tensors_.clear();
    tensor_shard_.clear();
    config_loaded_ = false;

    namespace fs = std::filesystem;

    std::string dir;
    std::vector<std::string> shard_paths;

    if (fs::is_directory(path)) {
        dir = path;
        // Collect all .safetensors files, sorted by name
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.path().extension() == ".safetensors") {
                shard_paths.push_back(entry.path().string());
            }
        }
        std::sort(shard_paths.begin(), shard_paths.end());
    } else if (fs::is_regular_file(path)) {
        dir = fs::path(path).parent_path().string();
        if (dir.empty()) dir = ".";

        // Check for multi-shard pattern: model-00001-of-00002.safetensors
        std::string fname = fs::path(path).filename().string();
        std::regex shard_pattern(R"((.+)-(\d{5})-of-(\d{5})\.safetensors)");
        std::smatch match;
        if (std::regex_match(fname, match, shard_pattern)) {
            std::string prefix = match[1].str();
            int total = std::stoi(match[3].str());
            for (int i = 1; i <= total; i++) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%s-%05d-of-%05d.safetensors",
                         prefix.c_str(), i, total);
                std::string shard_path = dir + "/" + buf;
                if (fs::exists(shard_path)) {
                    shard_paths.push_back(shard_path);
                }
            }
        } else {
            shard_paths.push_back(path);
        }
    } else {
        return false;
    }

    if (shard_paths.empty()) return false;

    shards_.resize(shard_paths.size());
    for (size_t i = 0; i < shard_paths.size(); i++) {
        if (!parse_shard(i, shard_paths[i])) {
            close_all();
            return false;
        }
    }

    // Load config.json from the model directory
    if (!load_config_json(dir)) {
        close_all();
        return false;
    }

    return true;
}

bool SafeTensorsReader::parse_shard(size_t shard_idx, const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    shards_[shard_idx].fd = fd;
    shards_[shard_idx].path = path;

    // Read 8-byte LE header size
    uint64_t header_size = 0;
    if (::pread(fd, &header_size, 8, 0) != 8) return false;

    // Sanity check: header shouldn't be larger than 100MB
    if (header_size > 100 * 1024 * 1024) return false;

    // Read JSON header
    std::vector<char> header_buf(header_size);
    ssize_t bytes_read = ::pread(fd, header_buf.data(),
                                  static_cast<size_t>(header_size), 8);
    if (bytes_read != static_cast<ssize_t>(header_size)) return false;

    shards_[shard_idx].data_start = 8 + header_size;

    // Parse JSON
    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_buf.begin(), header_buf.end());
    } catch (const nlohmann::json::parse_error&) {
        return false;
    }

    // Extract tensor info
    for (auto& [key, val] : header.items()) {
        if (key == "__metadata__") continue;

        TensorInfo info;
        info.name = key;

        // dtype
        std::string dtype_str = val.at("dtype").get<std::string>();
        if (dtype_str == "F16" || dtype_str == "float16") {
            info.dtype = "F16";
        } else if (dtype_str == "F32" || dtype_str == "float32") {
            info.dtype = "F32";
        } else if (dtype_str == "BF16" || dtype_str == "bfloat16") {
            info.dtype = "BF16";
        } else {
            info.dtype = dtype_str;
        }

        // shape
        for (const auto& s : val.at("shape")) {
            info.shape.push_back(s.get<int64_t>());
        }

        // data_offsets: [begin, end]
        auto offsets = val.at("data_offsets");
        info.begin = offsets[0].get<uint64_t>();
        info.end = offsets[1].get<uint64_t>();

        tensors_[key] = info;
        tensor_shard_[key] = shard_idx;
    }

    return true;
}

bool SafeTensorsReader::load_config_json(const std::string& dir) {
    std::string config_path = dir + "/config.json";
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) return false;

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error&) {
        return false;
    }

    // Map HuggingFace config.json fields to ModelConfig
    if (j.contains("model_type")) {
        config_.architecture = j["model_type"].get<std::string>();
    }
    if (j.contains("vocab_size")) {
        config_.vocab_size = j["vocab_size"].get<uint32_t>();
    }
    if (j.contains("hidden_size")) {
        config_.hidden_dim = j["hidden_size"].get<uint32_t>();
    }
    if (j.contains("intermediate_size")) {
        config_.intermediate_dim = j["intermediate_size"].get<uint32_t>();
    }
    if (j.contains("num_hidden_layers")) {
        config_.n_layers = j["num_hidden_layers"].get<uint32_t>();
    }
    if (j.contains("num_attention_heads")) {
        config_.n_heads = j["num_attention_heads"].get<uint32_t>();
    }
    if (j.contains("num_key_value_heads")) {
        config_.n_kv_heads = j["num_key_value_heads"].get<uint32_t>();
    } else {
        config_.n_kv_heads = config_.n_heads;  // default to MHA
    }
    if (j.contains("max_position_embeddings")) {
        config_.max_seq_len = j["max_position_embeddings"].get<uint32_t>();
    }
    if (j.contains("rope_theta")) {
        config_.rope_theta = j["rope_theta"].get<float>();
    }
    if (j.contains("rms_norm_eps")) {
        config_.norm_eps = j["rms_norm_eps"].get<float>();
    }

    // Compute head_dim
    if (config_.n_heads > 0) {
        config_.head_dim = config_.hidden_dim / config_.n_heads;
    }

    // Derive attention type
    config_.attention_type = derive_attention_type(config_.n_heads, config_.n_kv_heads);

    config_loaded_ = true;
    return true;
}

ModelConfig SafeTensorsReader::config() const {
    return config_;
}

const TensorInfo* SafeTensorsReader::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

bool SafeTensorsReader::read_tensor(const TensorInfo& info, void* buf, size_t buf_size) {
    uint64_t data_size = info.end - info.begin;
    if (buf_size < data_size) return false;

    auto shard_it = tensor_shard_.find(info.name);
    if (shard_it == tensor_shard_.end()) return false;

    const Shard& shard = shards_[shard_it->second];
    uint64_t file_offset = shard.data_start + info.begin;

    ssize_t bytes_read = ::pread(shard.fd, buf, static_cast<size_t>(data_size),
                                  static_cast<off_t>(file_offset));
    return bytes_read == static_cast<ssize_t>(data_size);
}

bool SafeTensorsReader::read_tensor_rows(const TensorInfo& info,
                                          int64_t row_start, int64_t row_count,
                                          void* buf, size_t buf_size) {
    if (info.shape.size() < 2) return false;

    int64_t cols = info.shape[1];
    size_t elem_size = info.element_size();
    if (elem_size == 0) return false;

    size_t row_bytes = static_cast<size_t>(cols) * elem_size;
    size_t total_bytes = static_cast<size_t>(row_count) * row_bytes;
    if (buf_size < total_bytes) return false;

    auto shard_it = tensor_shard_.find(info.name);
    if (shard_it == tensor_shard_.end()) return false;

    const Shard& shard = shards_[shard_it->second];
    uint64_t file_offset = shard.data_start + info.begin
                         + static_cast<uint64_t>(row_start) * row_bytes;

    ssize_t bytes_read = ::pread(shard.fd, buf, total_bytes,
                                  static_cast<off_t>(file_offset));
    return bytes_read == static_cast<ssize_t>(total_bytes);
}

std::vector<std::string> SafeTensorsReader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [k, v] : tensors_) {
        names.push_back(k);
    }
    return names;
}

void SafeTensorsReader::close_all() {
    for (auto& shard : shards_) {
        if (shard.fd >= 0) {
            ::close(shard.fd);
            shard.fd = -1;
        }
    }
    shards_.clear();
}

}  // namespace nos
