/// @file shared_metrics.cpp
/// @brief POSIX shared memory metrics transport implementation.

#include "server/shared_metrics.h"

#include <chrono>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Apple Clang does not support std::atomic_ref (C++20 P0019).
// Use compiler intrinsics for memory ordering on the seqlock sequence field.
#if defined(__clang__) || defined(__GNUC__)
#define NOS_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define NOS_LOAD_ACQUIRE(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#else
#include <atomic>
#define NOS_STORE_RELEASE(ptr, val) \
    std::atomic_ref<uint64_t>(*ptr).store(val, std::memory_order_release)
#define NOS_LOAD_ACQUIRE(ptr) \
    std::atomic_ref<uint64_t>(*ptr).load(std::memory_order_acquire)
#endif

namespace nos {

// ── MetricsWriter ───────────────────────────────────────────────────────────

MetricsWriter::MetricsWriter(const std::string& shm_name)
    : shm_name_(shm_name)
{
    fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd_ < 0) return;

    if (ftruncate(fd_, static_cast<off_t>(sizeof(SharedMetrics))) != 0) {
        close(fd_);
        fd_ = -1;
        return;
    }

    mapped_ = mmap(nullptr, sizeof(SharedMetrics),
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        close(fd_);
        fd_ = -1;
        return;
    }

    // Zero-initialize the shared region
    std::memset(mapped_, 0, sizeof(SharedMetrics));

    // Write initial magic
    auto* sm = static_cast<SharedMetrics*>(mapped_);
    sm->magic = 0x4E4F534D;
}

MetricsWriter::~MetricsWriter() {
    if (mapped_ != nullptr) {
        munmap(mapped_, sizeof(SharedMetrics));
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    if (!shm_name_.empty()) {
        shm_unlink(shm_name_.c_str());
    }
}

void MetricsWriter::update(const SharedMetrics& m) {
    if (mapped_ == nullptr) return;

    auto* sm = static_cast<SharedMetrics*>(mapped_);

    // Seqlock write: increment sequence (make odd), write, increment (make even)
    uint64_t seq = sm->sequence;
    seq++;
    NOS_STORE_RELEASE(&sm->sequence, seq);

    // Copy payload (skip the sequence field itself -- we manage it manually)
    SharedMetrics copy = m;
    copy.sequence = seq;
    copy.magic = 0x4E4F534D;

    // memcpy the payload portion (after magic and sequence)
    constexpr size_t header_size = sizeof(uint64_t) * 2;  // magic + sequence
    std::memcpy(static_cast<char*>(mapped_) + header_size,
                reinterpret_cast<const char*>(&copy) + header_size,
                sizeof(SharedMetrics) - header_size);

    // Finalize: increment sequence again (make even = consistent)
    seq++;
    NOS_STORE_RELEASE(&sm->sequence, seq);
}

bool MetricsWriter::is_open() const {
    return mapped_ != nullptr;
}

const std::string& MetricsWriter::shm_name() const {
    return shm_name_;
}

// ── MetricsReader ───────────────────────────────────────────────────────────

MetricsReader::MetricsReader(const std::string& shm_name)
    : shm_name_(shm_name)
{
    fd_ = shm_open(shm_name_.c_str(), O_RDONLY, 0);
    if (fd_ < 0) return;

    mapped_ = mmap(nullptr, sizeof(SharedMetrics),
                   PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        close(fd_);
        fd_ = -1;
    }
}

MetricsReader::~MetricsReader() {
    if (mapped_ != nullptr) {
        munmap(const_cast<void*>(mapped_), sizeof(SharedMetrics));
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    // Reader does NOT shm_unlink -- writer owns lifecycle
}

SharedMetrics MetricsReader::read() const {
    SharedMetrics result{};
    if (mapped_ == nullptr) return result;

    const auto* sm = static_cast<const SharedMetrics*>(mapped_);

    // Seqlock read loop: copy, check sequence consistency
    for (int retry = 0; retry < 1000; ++retry) {
        // Need const_cast because NOS_LOAD_ACQUIRE takes a non-const pointer;
        // the underlying __atomic_load_n is safe on const data.
        uint64_t seq1 = NOS_LOAD_ACQUIRE(
            const_cast<uint64_t*>(&sm->sequence));

        // If sequence is odd, writer is mid-update -- spin
        if ((seq1 & 1U) != 0) continue;

        // Copy the entire struct
        std::memcpy(&result, sm, sizeof(SharedMetrics));

        uint64_t seq2 = NOS_LOAD_ACQUIRE(
            const_cast<uint64_t*>(&sm->sequence));

        if (seq1 == seq2) {
            return result;  // Consistent read
        }
        // Mismatch: torn read, retry
    }

    // Fallback: return whatever we got (best effort after 1000 retries)
    return result;
}

bool MetricsReader::is_valid() const {
    if (mapped_ == nullptr) return false;

    SharedMetrics m = read();
    if (m.magic != 0x4E4F534D) return false;

    // Check freshness: last_update_epoch within 30 seconds
    auto now = std::chrono::system_clock::now();
    double now_epoch = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count()) / 1000.0;

    return (now_epoch - m.last_update_epoch) < 30.0;
}

bool MetricsReader::is_open() const {
    return mapped_ != nullptr;
}

}  // namespace nos
