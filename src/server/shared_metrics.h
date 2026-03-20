#pragma once

/// @file shared_metrics.h
/// @brief POSIX shared memory metrics transport with seqlock torn-read protection.
///
/// SharedMetrics is a fixed-size, memcpy-safe struct published by the server
/// (MetricsWriter) and consumed by the dashboard (MetricsReader) over a POSIX
/// shared memory segment. The seqlock pattern guarantees consistent reads without
/// blocking the writer.

#include <cstdint>
#include <cstring>
#include <string>

namespace nos {

/// Fixed-size metrics payload written to shared memory.
/// No pointers -- entirely memcpy-safe and mmap-friendly.
struct alignas(64) SharedMetrics {
    uint64_t magic = 0x4E4F534D;       ///< "NOSM" magic for validation
    uint64_t sequence = 0;             ///< Monotonic counter for seqlock

    // ── Performance ────────────────────────────────────────────────
    double tok_per_sec = 0.0;
    double ttft_ms = 0.0;
    double latency_p50_ms = 0.0;
    double latency_p95_ms = 0.0;
    double latency_p99_ms = 0.0;

    // ── Cache ──────────────────────────────────────────────────────
    double cache_hit_rate = 0.0;
    uint64_t evictions = 0;
    uint32_t resident_experts = 0;

    // ── Prefetch ───────────────────────────────────────────────────
    double oracle_rwp = 0.0;
    double waste_ratio = 0.0;
    char prefetch_mode[16] = {};

    // ── Routing ────────────────────────────────────────────────────
    double switch_rate = 0.0;
    double sticky_pct = 0.0;
    uint32_t shift_detections = 0;

    // ── Sparkline history ──────────────────────────────────────────
    float tok_per_sec_history[120] = {};  ///< 120 samples at 2Hz = 60 seconds
    uint32_t history_write_idx = 0;

    // ── Timestamp ──────────────────────────────────────────────────
    double last_update_epoch = 0.0;

    // ── Active slots ───────────────────────────────────────────────
    uint32_t active_slots = 0;
    uint32_t max_slots = 0;

    // ── Padding to cache-line alignment (struct is already alignas(64)) ──
    char _padding[4] = {};
};

static_assert(sizeof(SharedMetrics) < 4096,
              "SharedMetrics must fit in a single page");

/// Writer side: creates and owns the shared memory segment.
/// Uses seqlock pattern: increment sequence before and after memcpy.
class MetricsWriter {
public:
    /// Construct writer with explicit shm name.
    /// @param shm_name  POSIX shm name (must start with '/'), e.g. "/neuralos_metrics_12345"
    explicit MetricsWriter(const std::string& shm_name);
    ~MetricsWriter();

    MetricsWriter(const MetricsWriter&) = delete;
    MetricsWriter& operator=(const MetricsWriter&) = delete;

    /// Write metrics with seqlock protection.
    /// Increments sequence before and after write for torn-read detection.
    void update(const SharedMetrics& m);

    /// @return true if the shared memory segment was opened successfully.
    bool is_open() const;

    /// @return the shm name used (for discovery).
    const std::string& shm_name() const;

private:
    std::string shm_name_;
    int fd_ = -1;
    void* mapped_ = nullptr;
};

/// Reader side: opens existing shared memory segment (read-only).
/// Uses seqlock read loop to guarantee a consistent snapshot.
class MetricsReader {
public:
    /// Construct reader for an existing shm segment.
    /// @param shm_name  POSIX shm name (must start with '/')
    explicit MetricsReader(const std::string& shm_name);
    ~MetricsReader();

    MetricsReader(const MetricsReader&) = delete;
    MetricsReader& operator=(const MetricsReader&) = delete;

    /// Read a consistent SharedMetrics snapshot via seqlock retry loop.
    SharedMetrics read() const;

    /// @return true if magic matches and last_update_epoch is within 30 seconds.
    bool is_valid() const;

    /// @return true if the shared memory segment was opened successfully.
    bool is_open() const;

private:
    std::string shm_name_;
    int fd_ = -1;
    const void* mapped_ = nullptr;
};

}  // namespace nos
