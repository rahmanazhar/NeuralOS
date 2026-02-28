/// @file test_gguf_reader.cpp
/// @brief Tests for GGUF format reader.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "converter/gguf_reader.h"

namespace fs = std::filesystem;

/// Helper to write a minimal GGUF v3 file with metadata and one F32 tensor.
///
/// GGUF layout:
///   [4B magic "GGUF"] [4B version] [8B n_tensors] [8B n_kv]
///   [metadata key-value pairs]
///   [tensor info entries]
///   [alignment padding]
///   [tensor data]
class GgufBuilder {
public:
    GgufBuilder() = default;

    void add_string(const std::string& key, const std::string& value) {
        entries_.push_back({key, 8, value, 0, 0.0});  // type 8 = STRING
    }

    void add_uint32(const std::string& key, uint32_t value) {
        entries_.push_back({key, 4, "", value, 0.0});  // type 4 = UINT32
    }

    void add_float32(const std::string& key, float value) {
        entries_.push_back({key, 6, "", 0, static_cast<double>(value)});  // type 6 = FLOAT32
    }

    void add_tensor(const std::string& name, const std::vector<int64_t>& shape,
                     uint32_t type, const void* data, size_t data_size) {
        tensors_.push_back({name, shape, type, data, data_size});
    }

    bool write(const std::string& path) {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) return false;

        // Magic
        ofs.write("GGUF", 4);
        // Version
        uint32_t version = 3;
        ofs.write(reinterpret_cast<const char*>(&version), 4);
        // n_tensors
        uint64_t n_tensors = tensors_.size();
        ofs.write(reinterpret_cast<const char*>(&n_tensors), 8);
        // n_kv
        uint64_t n_kv = entries_.size();
        ofs.write(reinterpret_cast<const char*>(&n_kv), 8);

        // Write metadata key-value pairs
        for (const auto& e : entries_) {
            write_string(ofs, e.key);
            uint32_t type = e.type;
            ofs.write(reinterpret_cast<const char*>(&type), 4);

            if (e.type == 8) {  // STRING
                write_string(ofs, e.str_val);
            } else if (e.type == 4) {  // UINT32
                uint32_t v = static_cast<uint32_t>(e.uint_val);
                ofs.write(reinterpret_cast<const char*>(&v), 4);
            } else if (e.type == 6) {  // FLOAT32
                float v = static_cast<float>(e.float_val);
                ofs.write(reinterpret_cast<const char*>(&v), 4);
            }
        }

        // Compute tensor data offsets (relative to data section start)
        uint64_t data_offset = 0;
        std::vector<uint64_t> tensor_offsets;
        for (const auto& t : tensors_) {
            // Align to 32 bytes
            data_offset = (data_offset + 31) & ~static_cast<uint64_t>(31);
            tensor_offsets.push_back(data_offset);
            data_offset += t.data_size;
        }

        // Write tensor info entries
        for (size_t i = 0; i < tensors_.size(); i++) {
            const auto& t = tensors_[i];
            write_string(ofs, t.name);
            uint32_t n_dims = static_cast<uint32_t>(t.shape.size());
            ofs.write(reinterpret_cast<const char*>(&n_dims), 4);
            for (auto d : t.shape) {
                uint64_t dim = static_cast<uint64_t>(d);
                ofs.write(reinterpret_cast<const char*>(&dim), 8);
            }
            ofs.write(reinterpret_cast<const char*>(&t.type), 4);
            ofs.write(reinterpret_cast<const char*>(&tensor_offsets[i]), 8);
        }

        // Pad to alignment (32 bytes)
        auto pos = ofs.tellp();
        uint64_t aligned = (static_cast<uint64_t>(pos) + 31) & ~static_cast<uint64_t>(31);
        uint64_t pad = aligned - static_cast<uint64_t>(pos);
        std::vector<char> zeros(pad, 0);
        ofs.write(zeros.data(), static_cast<std::streamsize>(pad));

        // Write tensor data
        uint64_t data_written = 0;
        for (size_t i = 0; i < tensors_.size(); i++) {
            // Align
            uint64_t aligned_pos = (data_written + 31) & ~static_cast<uint64_t>(31);
            uint64_t gap = aligned_pos - data_written;
            if (gap > 0) {
                std::vector<char> pad_bytes(gap, 0);
                ofs.write(pad_bytes.data(), static_cast<std::streamsize>(gap));
                data_written += gap;
            }

            ofs.write(reinterpret_cast<const char*>(tensors_[i].data),
                      static_cast<std::streamsize>(tensors_[i].data_size));
            data_written += tensors_[i].data_size;
        }

        return true;
    }

private:
    struct Entry {
        std::string key;
        uint32_t type;
        std::string str_val;
        uint64_t uint_val;
        double float_val;
    };

    struct TensorEntry {
        std::string name;
        std::vector<int64_t> shape;
        uint32_t type;
        const void* data;
        size_t data_size;
    };

    std::vector<Entry> entries_;
    std::vector<TensorEntry> tensors_;

    static void write_string(std::ofstream& ofs, const std::string& s) {
        uint64_t len = s.size();
        ofs.write(reinterpret_cast<const char*>(&len), 8);
        ofs.write(s.data(), static_cast<std::streamsize>(len));
    }
};

TEST_CASE("GGUF: parse minimal file with metadata and tensor", "[gguf]") {
    auto tmpdir = fs::temp_directory_path() / "test_gguf_parse";
    fs::create_directories(tmpdir);
    std::string path = tmpdir.string() + "/model.gguf";

    // Create test data: 4x8 float matrix
    std::vector<float> data(32);
    for (int i = 0; i < 32; i++) data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.5f;

    GgufBuilder builder;
    builder.add_string("general.architecture", "llama");
    builder.add_uint32("llama.embedding_length", 64);
    builder.add_uint32("llama.feed_forward_length", 128);
    builder.add_uint32("llama.block_count", 2);
    builder.add_uint32("llama.attention.head_count", 4);
    builder.add_uint32("llama.attention.head_count_kv", 4);
    builder.add_uint32("llama.context_length", 2048);
    builder.add_float32("llama.rope.freq_base", 10000.0f);
    builder.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5f);

    builder.add_tensor("blk.0.attn_q.weight", {4, 8}, 0 /*F32*/,
                        data.data(), data.size() * sizeof(float));

    REQUIRE(builder.write(path));

    nos::GgufReader reader;
    REQUIRE(reader.open(path));

    SECTION("metadata extraction") {
        auto cfg = reader.config();
        CHECK(cfg.architecture == "llama");
        CHECK(cfg.hidden_dim == 64);
        CHECK(cfg.intermediate_dim == 128);
        CHECK(cfg.n_layers == 2);
        CHECK(cfg.n_heads == 4);
        CHECK(cfg.n_kv_heads == 4);
        CHECK(cfg.max_seq_len == 2048);
        CHECK(cfg.head_dim == 16);
        CHECK(cfg.attention_type == "mha");
    }

    SECTION("tensor names") {
        auto names = reader.tensor_names();
        REQUIRE(names.size() == 1);
        CHECK(names[0] == "blk.0.attn_q.weight");
    }

    SECTION("find and read tensor") {
        auto* info = reader.find_tensor("blk.0.attn_q.weight");
        REQUIRE(info != nullptr);
        CHECK(info->dtype == "F32");
        CHECK(info->shape[0] == 4);
        CHECK(info->shape[1] == 8);

        std::vector<float> buf(32);
        REQUIRE(reader.read_tensor(*info, buf.data(), buf.size() * sizeof(float)));

        for (int i = 0; i < 32; i++) {
            CHECK(buf[static_cast<size_t>(i)] == data[static_cast<size_t>(i)]);
        }
    }

    SECTION("missing tensor returns nullptr") {
        CHECK(reader.find_tensor("nonexistent") == nullptr);
    }

    SECTION("metadata getters") {
        CHECK(reader.get_metadata_string("general.architecture") == "llama");
        CHECK(reader.get_metadata_uint32("llama.block_count") == 2);
    }

    fs::remove_all(tmpdir);
}

TEST_CASE("GGUF: error cases", "[gguf]") {
    SECTION("nonexistent file") {
        nos::GgufReader reader;
        CHECK_FALSE(reader.open("/nonexistent/model.gguf"));
    }

    SECTION("bad magic") {
        auto tmpdir = fs::temp_directory_path() / "test_gguf_badmagic";
        fs::create_directories(tmpdir);
        std::string path = tmpdir.string() + "/bad.gguf";

        std::ofstream ofs(path, std::ios::binary);
        ofs.write("XXXX", 4);  // wrong magic
        uint32_t version = 3;
        ofs.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t zero = 0;
        ofs.write(reinterpret_cast<const char*>(&zero), 8);  // n_tensors
        ofs.write(reinterpret_cast<const char*>(&zero), 8);  // n_kv
        ofs.close();

        nos::GgufReader reader;
        CHECK_FALSE(reader.open(path));

        fs::remove_all(tmpdir);
    }

    SECTION("unsupported version") {
        auto tmpdir = fs::temp_directory_path() / "test_gguf_badver";
        fs::create_directories(tmpdir);
        std::string path = tmpdir.string() + "/badver.gguf";

        std::ofstream ofs(path, std::ios::binary);
        ofs.write("GGUF", 4);
        uint32_t version = 99;  // unsupported
        ofs.write(reinterpret_cast<const char*>(&version), 4);
        uint64_t zero = 0;
        ofs.write(reinterpret_cast<const char*>(&zero), 8);
        ofs.write(reinterpret_cast<const char*>(&zero), 8);
        ofs.close();

        nos::GgufReader reader;
        CHECK_FALSE(reader.open(path));

        fs::remove_all(tmpdir);
    }
}

TEST_CASE("GGUF: GQA config", "[gguf]") {
    auto tmpdir = fs::temp_directory_path() / "test_gguf_gqa";
    fs::create_directories(tmpdir);
    std::string path = tmpdir.string() + "/gqa.gguf";

    GgufBuilder builder;
    builder.add_string("general.architecture", "llama");
    builder.add_uint32("llama.embedding_length", 4096);
    builder.add_uint32("llama.feed_forward_length", 11008);
    builder.add_uint32("llama.block_count", 32);
    builder.add_uint32("llama.attention.head_count", 32);
    builder.add_uint32("llama.attention.head_count_kv", 8);
    builder.add_uint32("llama.context_length", 4096);

    REQUIRE(builder.write(path));

    nos::GgufReader reader;
    REQUIRE(reader.open(path));

    auto cfg = reader.config();
    CHECK(cfg.n_heads == 32);
    CHECK(cfg.n_kv_heads == 8);
    CHECK(cfg.attention_type == "gqa");
    CHECK(cfg.head_dim == 128);

    fs::remove_all(tmpdir);
}
