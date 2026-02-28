#pragma once

/// @file gguf_reader.h
/// @brief GGUF format reader implementing ModelReader interface.
///
/// Parses the GGUF binary format (v2/v3): validates magic/version,
/// extracts metadata key-values, reads tensor info and data.

#include <map>
#include <string>
#include <vector>

#include "converter/model_reader.h"

namespace nos {

class GgufReader : public ModelReader {
public:
    GgufReader();
    ~GgufReader() override;

    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    bool open(const std::string& path) override;
    ModelConfig config() const override;
    const TensorInfo* find_tensor(const std::string& name) const override;
    bool read_tensor(const TensorInfo& info, void* buf, size_t buf_size) override;
    bool read_tensor_rows(const TensorInfo& info,
                          int64_t row_start, int64_t row_count,
                          void* buf, size_t buf_size) override;
    std::vector<std::string> tensor_names() const override;

    /// Get a metadata string value by key.
    std::string get_metadata_string(const std::string& key) const;

    /// Get a metadata uint32 value by key.
    uint32_t get_metadata_uint32(const std::string& key, uint32_t default_val = 0) const;

    /// Get a metadata float value by key.
    float get_metadata_float(const std::string& key, float default_val = 0.0f) const;

private:
    /// GGUF value types
    enum class GgufType : uint32_t {
        UINT8   = 0,
        INT8    = 1,
        UINT16  = 2,
        INT16   = 3,
        UINT32  = 4,
        INT32   = 5,
        FLOAT32 = 6,
        BOOL    = 7,
        STRING  = 8,
        ARRAY   = 9,
        UINT64  = 10,
        INT64   = 11,
        FLOAT64 = 12,
    };

    /// GGUF tensor types
    enum class GgufTensorType : uint32_t {
        F32  = 0,
        F16  = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
    };

    struct MetaValue {
        GgufType type = GgufType::UINT32;
        uint64_t uint_val = 0;
        int64_t int_val = 0;
        double float_val = 0.0;
        std::string str_val;
        bool bool_val = false;
    };

    int fd_ = -1;
    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;  // start of tensor data section
    uint32_t alignment_ = 32;   // GGUF alignment (default 32)

    std::map<std::string, MetaValue> metadata_;
    std::map<std::string, TensorInfo> tensors_;
    ModelConfig config_{};

    bool parse_header();
    bool parse_metadata(uint64_t& offset, uint64_t n_kv);
    bool parse_tensor_infos(uint64_t& offset, uint64_t n_tensors);
    bool read_string(uint64_t& offset, std::string& out);
    bool read_value(uint64_t& offset, GgufType type, MetaValue& out);
    bool skip_value(uint64_t& offset, GgufType type);
    void extract_config();
    void close();

    template<typename T>
    bool read_at(uint64_t offset, T& val);
    bool read_bytes_at(uint64_t offset, void* buf, size_t len);

    /// Convert GGUF tensor type to our dtype string.
    static std::string gguf_type_to_dtype(GgufTensorType t);
    /// Bytes per element for a GGUF tensor type (approximate for quantized).
    static size_t gguf_type_element_size(GgufTensorType t);
};

}  // namespace nos
