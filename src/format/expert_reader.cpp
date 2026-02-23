/// @file expert_reader.cpp
/// @brief NXP expert file reader with CRC32C verification and retry on corruption.
///
/// Reads packed ternary expert weights and per-channel FP16 scale factors
/// from the .nxp binary format. CRC32C checksums are verified on read,
/// with configurable retry on integrity failure.

#include "expert_format.h"
#include "crc32.h"

#include <cstring>

#include <fcntl.h>
#include <unistd.h>

namespace nos {

NxpReader::NxpReader() = default;

NxpReader::~NxpReader() {
    close();
}

bool NxpReader::open(const std::string& path) {
    if (fd_ >= 0) {
        return false;  // Already open
    }

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        return false;
    }

    // Read header
    NxpFileHeader hdr{};
    if (::pread(fd_, &hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr))) {
        close();
        return false;
    }

    // Validate magic and version
    if (hdr.magic != NXP_MAGIC) {
        close();
        return false;
    }
    if (hdr.version != NXP_VERSION) {
        close();
        return false;
    }

    header_ = hdr;

    // Read expert index
    if (header_.total_experts > 0 && header_.index_offset > 0) {
        entries_.resize(static_cast<size_t>(header_.total_experts));
        auto index_size = static_cast<size_t>(header_.total_experts) * sizeof(NxpExpertEntry);
        ssize_t n = ::pread(fd_, entries_.data(), index_size,
                            static_cast<off_t>(header_.index_offset));
        if (n != static_cast<ssize_t>(index_size)) {
            entries_.clear();
            close();
            return false;
        }
    }

    return true;
}

const NxpFileHeader& NxpReader::header() const {
    return header_;
}

const NxpExpertEntry* NxpReader::find_expert(uint32_t layer_id, uint32_t expert_id) const {
    for (const auto& entry : entries_) {
        if (entry.layer_id == layer_id && entry.expert_id == expert_id) {
            return &entry;
        }
    }
    return nullptr;
}

int NxpReader::read_expert(const NxpExpertEntry& entry, uint8_t* buf, size_t buf_size,
                           int retry_count) {
    if (fd_ < 0) {
        return -1;
    }
    if (buf_size < entry.size) {
        return -1;  // Buffer too small
    }

    for (int attempt = 0; attempt <= retry_count; attempt++) {
        // Read weight data from file
        ssize_t n = ::pread(fd_, buf, static_cast<size_t>(entry.size),
                            static_cast<off_t>(entry.offset));
        if (n != static_cast<ssize_t>(entry.size)) {
            continue;  // Read error, retry
        }

        // Verify CRC32C
        uint32_t computed = crc32c(buf, static_cast<size_t>(entry.size));
        if (computed == entry.crc32) {
            return static_cast<int>(n);  // Success
        }
        // CRC mismatch -- retry
    }

    return -1;  // All retries exhausted
}

int NxpReader::read_scales(const NxpExpertEntry& entry, uint16_t* buf, size_t buf_size) {
    if (fd_ < 0) {
        return -1;
    }
    auto scale_bytes = static_cast<size_t>(entry.scale_size);
    if (buf_size < scale_bytes) {
        return -1;  // Buffer too small
    }

    ssize_t n = ::pread(fd_, buf, scale_bytes,
                        static_cast<off_t>(entry.scale_offset));
    if (n != static_cast<ssize_t>(scale_bytes)) {
        return -1;
    }
    return static_cast<int>(n);
}

void NxpReader::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    entries_.clear();
}

bool NxpReader::is_open() const {
    return fd_ >= 0;
}

size_t NxpReader::num_entries() const {
    return entries_.size();
}

}  // namespace nos
