/// @file expert_writer.cpp
/// @brief NXP expert file writer with 64-byte alignment and CRC32C checksums.
///
/// Writes packed ternary expert weights and per-channel FP16 scale factors
/// to the .nxp binary format. Weight data blocks are padded to NXP_ALIGNMENT
/// (64 bytes) for cache-line aligned reads.

#include "expert_format.h"
#include "crc32.h"

#include <cstring>

#include <fcntl.h>
#include <unistd.h>

namespace nos {

NxpWriter::NxpWriter() = default;

NxpWriter::~NxpWriter() {
    close();
}

bool NxpWriter::open(const std::string& path, const NxpFileHeader& header) {
    if (fd_ >= 0) {
        return false;  // Already open
    }

    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
        return false;
    }

    header_ = header;
    header_.magic = NXP_MAGIC;
    header_.version = NXP_VERSION;

    // Write placeholder header sequentially (will be rewritten in finalize via pwrite)
    write_offset_ = 0;
    write_bytes(reinterpret_cast<const uint8_t*>(&header_), sizeof(header_));
    entries_.clear();

#ifdef __linux__
    posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

    return true;
}

NxpExpertEntry NxpWriter::write_expert(uint32_t layer_id, uint32_t expert_id,
                                       const uint8_t* packed_weights, size_t weight_size,
                                       const uint16_t* scale_factors, uint32_t num_channels) {
    NxpExpertEntry entry{};
    entry.layer_id = layer_id;
    entry.expert_id = expert_id;

    // Pad to alignment before writing weights
    pad_to_alignment();

    // Record weight offset
    entry.offset = write_offset_;
    entry.size = static_cast<uint64_t>(weight_size);

    // Write packed weights
    if (weight_size > 0 && packed_weights != nullptr) {
        write_bytes(packed_weights, weight_size);
    }

    // Compute CRC32C of weight data
    if (weight_size > 0 && packed_weights != nullptr) {
        entry.crc32 = crc32c(packed_weights, weight_size);
    } else {
        entry.crc32 = crc32c(nullptr, 0);
    }

    // Pad to alignment before writing scale factors
    pad_to_alignment();

    // Write scale factors (FP16 as raw uint16_t array) -- SEPARATE from weights
    entry.scale_offset = write_offset_;
    auto scale_bytes = static_cast<size_t>(num_channels) * sizeof(uint16_t);
    entry.scale_size = static_cast<uint32_t>(scale_bytes);
    entry.num_channels = num_channels;

    if (scale_bytes > 0 && scale_factors != nullptr) {
        write_bytes(reinterpret_cast<const uint8_t*>(scale_factors), scale_bytes);
    }

    // Zero reserved field
    std::memset(entry.reserved, 0, sizeof(entry.reserved));

    entries_.push_back(entry);
    return entry;
}

bool NxpWriter::finalize() {
    if (fd_ < 0) {
        return false;
    }

    // Pad to alignment before writing index
    pad_to_alignment();

    // Record index offset
    uint64_t index_offset = write_offset_;

    // Write all expert entries as the index
    for (const auto& entry : entries_) {
        write_bytes(reinterpret_cast<const uint8_t*>(&entry), sizeof(entry));
    }

    // Update header with correct offsets
    header_.index_offset = index_offset;
    header_.data_offset = NXP_HEADER_SIZE;  // Data starts right after header
    header_.total_experts = entries_.size();

    // Seek to beginning and rewrite header
    if (!write_at(0, &header_, sizeof(header_))) {
        close();
        return false;
    }

    close();
    return true;
}

bool NxpWriter::is_open() const {
    return fd_ >= 0;
}

void NxpWriter::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void NxpWriter::write_bytes(const uint8_t* data, size_t len) {
    size_t written = 0;
    while (written < len) {
        auto n = ::write(fd_, data + written, len - written);
        if (n < 0) {
            break;  // Error
        }
        written += static_cast<size_t>(n);
    }
    write_offset_ += written;
}

bool NxpWriter::write_at(uint64_t offset, const void* data, size_t len) {
    ssize_t n = ::pwrite(fd_, data, len, static_cast<off_t>(offset));
    return n == static_cast<ssize_t>(len);
}

void NxpWriter::pad_to_alignment() {
    uint64_t remainder = write_offset_ % NXP_ALIGNMENT;
    if (remainder != 0) {
        auto padding = static_cast<size_t>(NXP_ALIGNMENT - remainder);
        uint8_t zeros[NXP_ALIGNMENT]{};
        write_bytes(zeros, padding);
    }
}

}  // namespace nos
