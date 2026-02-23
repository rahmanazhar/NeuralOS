#pragma once

/// @file platform_io.h
/// @brief Platform-abstracted async I/O layer (SQ/CQ model).
///
/// Mirrors the io_uring submission/completion queue model as the
/// platform-agnostic interface. All backends implement the same ring
/// interface: submit_read (non-blocking) + poll (blocking with timeout).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <vector>

namespace nos {

/// Completion result from an async I/O operation.
struct IoCompletion {
    int result;      ///< Bytes read, or negative errno on failure
    void* userdata;  ///< Caller-supplied context pointer
};

/// Abstract base for platform-specific I/O backends.
///
/// Implementations:
///   - PreadBackend: synchronous pread fallback (both platforms)
///   - IoUringBackend: Linux io_uring via liburing (Linux only)
class PlatformIO {
public:
    virtual ~PlatformIO() = default;

    /// Submit a read request.
    ///
    /// For synchronous backends (PreadBackend), the read completes immediately
    /// and the result is available on the next poll().
    ///
    /// @param fd        File descriptor to read from
    /// @param buf       Destination buffer
    /// @param len       Number of bytes to read
    /// @param offset    File offset to read from
    /// @param userdata  Caller context, returned in IoCompletion
    /// @return 0 on success, or negative errno on error
    virtual int submit_read(int fd, void* buf, size_t len, off_t offset, void* userdata) = 0;

    /// Poll for completed I/O operations.
    ///
    /// Blocks up to timeout_ms milliseconds.
    /// -1 = block forever, 0 = non-blocking.
    ///
    /// @param completions  Output array for completed operations
    /// @param max_events   Maximum completions to return
    /// @param timeout_ms   Maximum wait time in milliseconds (-1 for infinite)
    /// @return Number of completions harvested, or negative errno
    virtual int poll(IoCompletion* completions, int max_events, int timeout_ms) = 0;

    /// Number of in-flight (submitted but not yet polled) requests.
    virtual int pending() const = 0;

    /// Factory: create the best available I/O backend for the current platform.
    ///
    /// On Linux: creates IoUringBackend if io_uring is available, else PreadBackend.
    /// On macOS: creates PreadBackend (kqueue cannot async-complete file reads).
    static std::unique_ptr<PlatformIO> create();
};

/// Synchronous pread fallback backend (available on all POSIX platforms).
///
/// submit_read() executes pread() immediately and buffers the completion.
/// poll() drains the internal completion queue.
class PreadBackend : public PlatformIO {
public:
    int submit_read(int fd, void* buf, size_t len, off_t offset, void* userdata) override;
    int poll(IoCompletion* completions, int max_events, int timeout_ms) override;
    int pending() const override;

private:
    std::vector<IoCompletion> completions_;
};

}  // namespace nos
