/// @file pread_backend.cpp
/// @brief Synchronous pread fallback I/O backend and PlatformIO factory.
///
/// PreadBackend executes pread() synchronously on submit_read() and buffers
/// the result. poll() drains the internal completion queue. This backend
/// works on all POSIX platforms and serves as the macOS default.

#include "platform_io.h"

#include <algorithm>
#include <cerrno>
#include <unistd.h>

namespace nos {

// ---------------------------------------------------------------------------
// PreadBackend implementation
// ---------------------------------------------------------------------------

int PreadBackend::submit_read(int fd, void* buf, size_t len,
                              off_t offset, void* userdata) {
    // Execute the read synchronously right now.
    ssize_t bytes = ::pread(fd, buf, len, offset);

    IoCompletion c{};
    c.userdata = userdata;

    if (bytes < 0) {
        // pread failed -- store negative errno in the completion result.
        c.result = -errno;
    } else {
        c.result = static_cast<int>(bytes);
    }

    completions_.push_back(c);
    return 0;  // submit always succeeds (the error is in the completion)
}

int PreadBackend::poll(IoCompletion* completions, int max_events,
                       int /*timeout_ms*/) {
    // Drain the internal queue up to max_events.
    int count = std::min(static_cast<int>(completions_.size()), max_events);
    for (int i = 0; i < count; ++i) {
        completions[i] = completions_[static_cast<size_t>(i)];
    }

    // Remove the delivered completions from the front.
    if (count > 0) {
        completions_.erase(completions_.begin(),
                           completions_.begin() + static_cast<ptrdiff_t>(count));
    }

    return count;
}

int PreadBackend::pending() const {
    return static_cast<int>(completions_.size());
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<PlatformIO> PlatformIO::create() {
    // On macOS (and any non-Linux platform): always return PreadBackend.
    // On Linux: will try IoUringBackend first once Task 2 implements it.
    return std::make_unique<PreadBackend>();
}

}  // namespace nos
