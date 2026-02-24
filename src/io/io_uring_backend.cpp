/// @file io_uring_backend.cpp
/// @brief Linux io_uring async I/O backend via liburing.
///
/// Only compiled on Linux (guarded in CMakeLists.txt). Uses liburing for
/// async read operations with IORING_SETUP_COOP_TASKRUN optimization.
///
/// Implements the PlatformIO SQ/CQ interface:
///   submit_read -> io_uring_prep_read + io_uring_submit
///   poll        -> io_uring_wait_cqe / io_uring_for_each_cqe

#ifdef NOS_HAS_IO_URING

#include "platform_io.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <liburing.h>

namespace nos {

/// Queue depth for the io_uring ring. 64 is a reasonable default for
/// file I/O workloads (expert loading). Can be tuned later.
static constexpr int QUEUE_DEPTH = 64;

class IoUringBackend : public PlatformIO {
public:
    IoUringBackend() {
        // Try with COOP_TASKRUN first (kernel 5.19+, reduces overhead).
        struct io_uring_params params{};
        params.flags = IORING_SETUP_COOP_TASKRUN;

        int ret = io_uring_queue_init_params(QUEUE_DEPTH, &ring_, &params);
        if (ret < 0) {
            // Fall back to plain init for older kernels.
            std::memset(&params, 0, sizeof(params));
            ret = io_uring_queue_init(QUEUE_DEPTH, &ring_, 0);
            if (ret < 0) {
                throw std::runtime_error(
                    std::string("io_uring init failed: ") +
                    std::strerror(-ret));
            }
        }
    }

    ~IoUringBackend() override {
        io_uring_queue_exit(&ring_);
    }

    // Non-copyable, non-movable (ring_ is not trivially movable).
    IoUringBackend(const IoUringBackend&) = delete;
    IoUringBackend& operator=(const IoUringBackend&) = delete;

    int submit_read(int fd, void* buf, size_t len,
                    off_t offset, void* userdata) override {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            return -EBUSY;  // SQ ring is full.
        }

        io_uring_prep_read(sqe, fd, buf, static_cast<unsigned>(len), offset);
        io_uring_sqe_set_data(sqe, userdata);

        int ret = io_uring_submit(&ring_);
        if (ret < 0) {
            return ret;
        }

        ++pending_count_;
        return 0;
    }

    int poll(IoCompletion* completions, int max_events,
             int timeout_ms) override {
        if (pending_count_ == 0) {
            return 0;  // Nothing submitted, nothing to wait for.
        }

        struct io_uring_cqe* cqe = nullptr;
        int ret = 0;

        if (timeout_ms < 0) {
            // Block indefinitely for at least one completion.
            ret = io_uring_wait_cqe(&ring_, &cqe);
        } else {
            struct __kernel_timespec ts{};
            ts.tv_sec  = timeout_ms / 1000;
            ts.tv_nsec = static_cast<long long>(timeout_ms % 1000) * 1000000LL;
            ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
        }

        if (ret == -ETIME || ret == -EINTR) {
            return 0;  // Timeout or interrupted, no completions ready.
        }
        if (ret < 0) {
            return ret;  // Real error.
        }

        // Harvest all available CQEs up to max_events.
        int count = 0;
        unsigned head = 0;
        io_uring_for_each_cqe(&ring_, head, cqe) {
            if (count >= max_events) {
                break;
            }
            completions[count].result = cqe->res;
            completions[count].userdata = io_uring_cqe_get_data(cqe);
            ++count;
        }

        io_uring_cq_advance(&ring_, static_cast<unsigned>(count));
        pending_count_ -= count;

        return count;
    }

    int pending() const override {
        return pending_count_;
    }

private:
    struct io_uring ring_{};
    int pending_count_ = 0;
};

// ---------------------------------------------------------------------------
// Factory (Linux variant): try io_uring, fall back to pread.
// ---------------------------------------------------------------------------

std::unique_ptr<PlatformIO> PlatformIO::create() {
    try {
        auto backend = std::make_unique<IoUringBackend>();
        std::cerr << "[nos] I/O backend: io_uring" << std::endl;
        return backend;
    } catch (const std::runtime_error& e) {
        std::cerr << "[nos] io_uring unavailable (" << e.what()
                  << "), falling back to pread" << std::endl;
        return std::make_unique<PreadBackend>();
    }
}

}  // namespace nos

#endif  // NOS_HAS_IO_URING
