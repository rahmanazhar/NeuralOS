#pragma once

/// @file model_reader.h
/// @brief Abstract model reader interface for SafeTensors/GGUF format support.
///
/// Defines TensorInfo and the ModelReader base class. The factory method
/// ModelReader::create() auto-detects format by magic bytes.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "converter/model_config.h"

namespace nos {

/// Describes a single tensor in the model file.
struct TensorInfo {
    std::string name;
    std::string dtype;  // "F16", "F32", "BF16"
    std::vector<int64_t> shape;
    uint64_t begin = 0;  // byte offset from data section start
    uint64_t end = 0;    // byte offset from data section start

    /// Total number of elements.
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    /// Bytes per element based on dtype.
    size_t element_size() const {
        if (dtype == "F32") return 4;
        if (dtype == "F16" || dtype == "BF16") return 2;
        return 0;
    }
};

/// Abstract base class for reading model files (SafeTensors, GGUF).
class ModelReader {
public:
    virtual ~ModelReader() = default;

    /// Open and parse a model file (or directory for multi-shard SafeTensors).
    virtual bool open(const std::string& path) = 0;

    /// Return the model configuration extracted from the file.
    virtual ModelConfig config() const = 0;

    /// Find a tensor by name, returns nullptr if not found.
    virtual const TensorInfo* find_tensor(const std::string& name) const = 0;

    /// Read entire tensor data into buf. buf_size must be >= (end - begin).
    virtual bool read_tensor(const TensorInfo& info, void* buf, size_t buf_size) = 0;

    /// Read a row slice of a 2D tensor.
    /// @param row_start  First row index
    /// @param row_count  Number of rows to read
    virtual bool read_tensor_rows(const TensorInfo& info,
                                  int64_t row_start, int64_t row_count,
                                  void* buf, size_t buf_size) = 0;

    /// Return all tensor names in the model.
    virtual std::vector<std::string> tensor_names() const = 0;

    /// Factory: auto-detect format from magic bytes and return appropriate reader.
    static std::unique_ptr<ModelReader> create(const std::string& path);
};

}  // namespace nos
