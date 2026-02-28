/// @file gguf_reader.cpp
/// @brief GGUF format reader implementation.

#include "converter/gguf_reader.h"

#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include "kernel/packing.h"  // fp16_to_fp32

namespace nos {

GgufReader::GgufReader() = default;

GgufReader::~GgufReader() {
    close();
}

bool GgufReader::open(const std::string& path) {
    close();
    metadata_.clear();
    tensors_.clear();

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    if (!parse_header()) {
        close();
        return false;
    }

    extract_config();
    return true;
}

bool GgufReader::parse_header() {
    // Validate magic: "GGUF" (4 bytes)
    char magic[4];
    if (!read_bytes_at(0, magic, 4)) return false;
    if (std::memcmp(magic, "GGUF", 4) != 0) return false;

    // Version (uint32)
    if (!read_at(4, version_)) return false;
    if (version_ < 2 || version_ > 3) return false;

    // Number of tensors (uint64)
    uint64_t n_tensors = 0;
    if (!read_at(8, n_tensors)) return false;

    // Number of metadata key-value pairs (uint64)
    uint64_t n_kv = 0;
    if (!read_at(16, n_kv)) return false;

    uint64_t offset = 24;

    // Parse metadata
    if (!parse_metadata(offset, n_kv)) return false;

    // Check for alignment override in metadata
    auto align_it = metadata_.find("general.alignment");
    if (align_it != metadata_.end()) {
        alignment_ = static_cast<uint32_t>(align_it->second.uint_val);
        if (alignment_ == 0) alignment_ = 32;
    }

    // Parse tensor infos
    if (!parse_tensor_infos(offset, n_tensors)) return false;

    // Compute data section offset (aligned)
    data_offset_ = (offset + alignment_ - 1) & ~(static_cast<uint64_t>(alignment_) - 1);

    // Apply data_offset to all tensor offsets
    for (auto& [name, info] : tensors_) {
        // info.begin is relative to data section; convert to absolute
        info.begin += data_offset_;
        uint64_t data_size = info.end - (info.begin - data_offset_);
        info.end = info.begin + data_size;
    }

    return true;
}

bool GgufReader::parse_metadata(uint64_t& offset, uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key;
        if (!read_string(offset, key)) return false;

        uint32_t type_raw = 0;
        if (!read_at(offset, type_raw)) return false;
        offset += 4;

        auto type = static_cast<GgufType>(type_raw);
        MetaValue val;
        val.type = type;
        if (!read_value(offset, type, val)) return false;

        metadata_[key] = val;
    }
    return true;
}

bool GgufReader::parse_tensor_infos(uint64_t& offset, uint64_t n_tensors) {
    for (uint64_t i = 0; i < n_tensors; i++) {
        std::string name;
        if (!read_string(offset, name)) return false;

        uint32_t n_dims = 0;
        if (!read_at(offset, n_dims)) return false;
        offset += 4;

        std::vector<int64_t> shape(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = 0;
            if (!read_at(offset, dim)) return false;
            offset += 8;
            shape[d] = static_cast<int64_t>(dim);
        }

        uint32_t type_raw = 0;
        if (!read_at(offset, type_raw)) return false;
        offset += 4;

        uint64_t tensor_offset = 0;
        if (!read_at(offset, tensor_offset)) return false;
        offset += 8;

        auto ttype = static_cast<GgufTensorType>(type_raw);

        TensorInfo info;
        info.name = name;
        info.dtype = gguf_type_to_dtype(ttype);
        info.shape = shape;
        info.begin = tensor_offset;  // relative to data section for now

        // Compute size
        int64_t numel = 1;
        for (auto d : shape) numel *= d;
        size_t elem_size = gguf_type_element_size(ttype);
        uint64_t data_size = static_cast<uint64_t>(numel) * elem_size;

        // For quantized types, adjust size based on block structure
        if (ttype == GgufTensorType::Q8_0) {
            // Q8_0: blocks of 32 elements, each block = 2 bytes (scale) + 32 bytes (data) = 34 bytes
            uint64_t n_blocks = (static_cast<uint64_t>(numel) + 31) / 32;
            data_size = n_blocks * 34;
        }

        info.end = info.begin + data_size;
        tensors_[name] = info;
    }
    return true;
}

bool GgufReader::read_string(uint64_t& offset, std::string& out) {
    uint64_t len = 0;
    if (!read_at(offset, len)) return false;
    offset += 8;

    if (len > 64 * 1024 * 1024) return false;  // sanity check

    out.resize(static_cast<size_t>(len));
    if (len > 0) {
        if (!read_bytes_at(offset, out.data(), static_cast<size_t>(len))) return false;
    }
    offset += len;
    return true;
}

bool GgufReader::read_value(uint64_t& offset, GgufType type, MetaValue& out) {
    switch (type) {
        case GgufType::UINT8: {
            uint8_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 1;
            out.uint_val = v;
            return true;
        }
        case GgufType::INT8: {
            int8_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 1;
            out.int_val = v;
            return true;
        }
        case GgufType::UINT16: {
            uint16_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 2;
            out.uint_val = v;
            return true;
        }
        case GgufType::INT16: {
            int16_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 2;
            out.int_val = v;
            return true;
        }
        case GgufType::UINT32: {
            uint32_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 4;
            out.uint_val = v;
            return true;
        }
        case GgufType::INT32: {
            int32_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 4;
            out.int_val = v;
            return true;
        }
        case GgufType::FLOAT32: {
            float v = 0;
            if (!read_at(offset, v)) return false;
            offset += 4;
            out.float_val = v;
            return true;
        }
        case GgufType::BOOL: {
            uint8_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 1;
            out.bool_val = (v != 0);
            return true;
        }
        case GgufType::STRING: {
            return read_string(offset, out.str_val);
        }
        case GgufType::ARRAY: {
            uint32_t elem_type_raw = 0;
            if (!read_at(offset, elem_type_raw)) return false;
            offset += 4;
            uint64_t count = 0;
            if (!read_at(offset, count)) return false;
            offset += 8;
            auto elem_type = static_cast<GgufType>(elem_type_raw);
            for (uint64_t i = 0; i < count; i++) {
                if (!skip_value(offset, elem_type)) return false;
            }
            return true;
        }
        case GgufType::UINT64: {
            uint64_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 8;
            out.uint_val = v;
            return true;
        }
        case GgufType::INT64: {
            int64_t v = 0;
            if (!read_at(offset, v)) return false;
            offset += 8;
            out.int_val = v;
            return true;
        }
        case GgufType::FLOAT64: {
            double v = 0;
            if (!read_at(offset, v)) return false;
            offset += 8;
            out.float_val = v;
            return true;
        }
        default:
            return false;
    }
}

bool GgufReader::skip_value(uint64_t& offset, GgufType type) {
    MetaValue dummy;
    return read_value(offset, type, dummy);
}

void GgufReader::extract_config() {
    // Extract architecture
    auto arch_it = metadata_.find("general.architecture");
    if (arch_it != metadata_.end()) {
        config_.architecture = arch_it->second.str_val;
    }

    std::string arch = config_.architecture;

    // Map GGUF metadata keys to ModelConfig fields
    auto get_u32 = [&](const std::string& key) -> uint32_t {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return 0;
        return static_cast<uint32_t>(it->second.uint_val);
    };
    auto get_f32 = [&](const std::string& key, float def) -> float {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return def;
        return static_cast<float>(it->second.float_val);
    };

    config_.hidden_dim = get_u32(arch + ".embedding_length");
    config_.intermediate_dim = get_u32(arch + ".feed_forward_length");
    config_.n_layers = get_u32(arch + ".block_count");
    config_.n_heads = get_u32(arch + ".attention.head_count");
    config_.n_kv_heads = get_u32(arch + ".attention.head_count_kv");
    if (config_.n_kv_heads == 0) config_.n_kv_heads = config_.n_heads;
    config_.max_seq_len = get_u32(arch + ".context_length");
    if (config_.max_seq_len == 0) config_.max_seq_len = 2048;
    config_.vocab_size = get_u32(arch + ".vocab_size");
    if (config_.vocab_size == 0) {
        // Try general.vocab_size or tokenizer.ggml.tokens array length
        config_.vocab_size = get_u32("general.vocab_size");
    }

    config_.rope_theta = get_f32(arch + ".rope.freq_base", 10000.0f);
    config_.norm_eps = get_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);

    if (config_.n_heads > 0) {
        config_.head_dim = config_.hidden_dim / config_.n_heads;
    }
    config_.attention_type = derive_attention_type(config_.n_heads, config_.n_kv_heads);
}

ModelConfig GgufReader::config() const {
    return config_;
}

const TensorInfo* GgufReader::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

bool GgufReader::read_tensor(const TensorInfo& info, void* buf, size_t buf_size) {
    uint64_t data_size = info.end - info.begin;
    if (buf_size < data_size) return false;

    return read_bytes_at(info.begin, buf, static_cast<size_t>(data_size));
}

bool GgufReader::read_tensor_rows(const TensorInfo& info,
                                   int64_t row_start, int64_t row_count,
                                   void* buf, size_t buf_size) {
    if (info.shape.size() < 2) return false;

    int64_t cols = info.shape[1];
    size_t elem_size = info.element_size();
    if (elem_size == 0) return false;

    size_t row_bytes = static_cast<size_t>(cols) * elem_size;
    size_t total_bytes = static_cast<size_t>(row_count) * row_bytes;
    if (buf_size < total_bytes) return false;

    uint64_t file_offset = info.begin + static_cast<uint64_t>(row_start) * row_bytes;
    return read_bytes_at(file_offset, buf, total_bytes);
}

std::vector<std::string> GgufReader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [k, v] : tensors_) {
        names.push_back(k);
    }
    return names;
}

std::string GgufReader::get_metadata_string(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return "";
    return it->second.str_val;
}

uint32_t GgufReader::get_metadata_uint32(const std::string& key, uint32_t default_val) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return default_val;
    return static_cast<uint32_t>(it->second.uint_val);
}

float GgufReader::get_metadata_float(const std::string& key, float default_val) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return default_val;
    return static_cast<float>(it->second.float_val);
}

template<typename T>
bool GgufReader::read_at(uint64_t offset, T& val) {
    return ::pread(fd_, &val, sizeof(T), static_cast<off_t>(offset)) == sizeof(T);
}

bool GgufReader::read_bytes_at(uint64_t offset, void* buf, size_t len) {
    return ::pread(fd_, buf, len, static_cast<off_t>(offset)) == static_cast<ssize_t>(len);
}

std::string GgufReader::gguf_type_to_dtype(GgufTensorType t) {
    switch (t) {
        case GgufTensorType::F32:  return "F32";
        case GgufTensorType::F16:  return "F16";
        case GgufTensorType::Q8_0: return "Q8_0";
        default: return "UNKNOWN";
    }
}

size_t GgufReader::gguf_type_element_size(GgufTensorType t) {
    switch (t) {
        case GgufTensorType::F32:  return 4;
        case GgufTensorType::F16:  return 2;
        case GgufTensorType::Q8_0: return 1;  // approximate; actual size computed per block
        default: return 0;
    }
}

void GgufReader::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace nos
