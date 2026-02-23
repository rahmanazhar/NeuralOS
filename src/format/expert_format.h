#pragma once

/// @file expert_format.h
/// @brief NXP expert file format definitions and writer/reader classes.
///
/// The .nxp format stores packed ternary expert weights with 64-byte
/// cache-line alignment and CRC32C integrity checksums. Layout:
///   [256B header] [expert index entries] [64B-aligned weight blocks]

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nos {

/// Magic number: "NXP0" as little-endian uint32_t
constexpr uint32_t NXP_MAGIC = 0x4E585030;

/// Format version
constexpr uint32_t NXP_VERSION = 1;

/// Fixed header size (64-byte aligned)
constexpr size_t NXP_HEADER_SIZE = 256;

/// Cache-line alignment for weight data blocks
constexpr size_t NXP_ALIGNMENT = 64;

/// File header: fixed 256 bytes at the start of every .nxp file.
#pragma pack(push, 1)
struct NxpFileHeader {
    uint32_t magic;               ///< NXP_MAGIC (0x4E585030)
    uint32_t version;             ///< NXP_VERSION
    uint32_t num_layers;          ///< Total number of layers in the model
    uint32_t experts_per_layer;   ///< Number of experts per layer
    uint32_t hidden_dim;          ///< Model hidden dimension
    uint32_t intermediate_dim;    ///< Expert intermediate dimension
    uint32_t packing_mode;        ///< 0 = 5-per-byte, 1 = 4-per-byte
    uint32_t scale_dtype;         ///< 0 = FP16
    uint64_t index_offset;        ///< Byte offset to expert index array
    uint64_t data_offset;         ///< Byte offset to first expert weight data
    uint64_t total_experts;       ///< Total expert count (num_layers * experts_per_layer)
    uint8_t reserved[256 - 56];   ///< Padding to 256 bytes
};
#pragma pack(pop)

static_assert(sizeof(NxpFileHeader) == NXP_HEADER_SIZE,
              "NxpFileHeader must be exactly 256 bytes");

/// Expert index entry: 64 bytes per expert.
#pragma pack(push, 1)
struct NxpExpertEntry {
    uint32_t layer_id;        ///< Layer index
    uint32_t expert_id;       ///< Expert index within layer
    uint64_t offset;          ///< Byte offset from file start to weight data
    uint64_t size;            ///< Size of packed weight data in bytes
    uint32_t crc32;           ///< CRC32C checksum of weight data
    uint32_t num_channels;    ///< Number of channels (for scale factor array sizing)
    uint64_t scale_offset;    ///< Byte offset to per-channel FP16 scale factors
    uint32_t scale_size;      ///< Size of scale factor data in bytes
    uint8_t reserved[20];     ///< Padding to 64 bytes (44 bytes of fields + 20 reserved)
};
#pragma pack(pop)

static_assert(sizeof(NxpExpertEntry) == 64,
              "NxpExpertEntry must be exactly 64 bytes");

// -- NXP Writer --

/// Writes packed expert data to a .nxp file with 64-byte alignment and CRC32C.
class NxpWriter {
public:
    NxpWriter();
    ~NxpWriter();

    NxpWriter(const NxpWriter&) = delete;
    NxpWriter& operator=(const NxpWriter&) = delete;

    bool open(const std::string& path, const NxpFileHeader& header);

    NxpExpertEntry write_expert(uint32_t layer_id, uint32_t expert_id,
                                const uint8_t* packed_weights, size_t weight_size,
                                const uint16_t* scale_factors, uint32_t num_channels);

    bool finalize();
    bool is_open() const;

private:
    int fd_ = -1;
    uint64_t write_offset_ = 0;
    NxpFileHeader header_{};
    std::vector<NxpExpertEntry> entries_;

    void close();
    void write_bytes(const uint8_t* data, size_t len);
    bool write_at(uint64_t offset, const void* data, size_t len);
    void pad_to_alignment();
};

// -- NXP Reader --

/// Reads and verifies packed expert data from a .nxp file.
class NxpReader {
public:
    NxpReader();
    ~NxpReader();

    NxpReader(const NxpReader&) = delete;
    NxpReader& operator=(const NxpReader&) = delete;

    bool open(const std::string& path);
    const NxpFileHeader& header() const;
    const NxpExpertEntry* find_expert(uint32_t layer_id, uint32_t expert_id) const;

    int read_expert(const NxpExpertEntry& entry, uint8_t* buf, size_t buf_size,
                    int retry_count = 3);
    int read_scales(const NxpExpertEntry& entry, uint16_t* buf, size_t buf_size);

    void close();
    bool is_open() const;
    size_t num_entries() const;

private:
    int fd_ = -1;
    NxpFileHeader header_{};
    std::vector<NxpExpertEntry> entries_;
};

}  // namespace nos
