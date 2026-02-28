#pragma once

/// @file safetensors_reader.h
/// @brief SafeTensors format reader implementing ModelReader interface.
///
/// Parses the SafeTensors binary format: 8-byte LE header_size + JSON header
/// + raw tensor data. Supports multi-shard models.

#include <map>
#include <string>
#include <vector>

#include "converter/model_reader.h"

namespace nos {

class SafeTensorsReader : public ModelReader {
public:
    SafeTensorsReader();
    ~SafeTensorsReader() override;

    SafeTensorsReader(const SafeTensorsReader&) = delete;
    SafeTensorsReader& operator=(const SafeTensorsReader&) = delete;

    bool open(const std::string& path) override;
    ModelConfig config() const override;
    const TensorInfo* find_tensor(const std::string& name) const override;
    bool read_tensor(const TensorInfo& info, void* buf, size_t buf_size) override;
    bool read_tensor_rows(const TensorInfo& info,
                          int64_t row_start, int64_t row_count,
                          void* buf, size_t buf_size) override;
    std::vector<std::string> tensor_names() const override;

private:
    /// Represents a single .safetensors shard file.
    struct Shard {
        int fd = -1;
        uint64_t data_start = 0;  // offset to tensor data region
        std::string path;
    };

    std::vector<Shard> shards_;
    std::map<std::string, TensorInfo> tensors_;
    std::map<std::string, size_t> tensor_shard_;  // tensor name -> shard index
    ModelConfig config_{};
    bool config_loaded_ = false;

    bool parse_shard(size_t shard_idx, const std::string& path);
    bool load_config_json(const std::string& dir);
    void close_all();
};

}  // namespace nos
