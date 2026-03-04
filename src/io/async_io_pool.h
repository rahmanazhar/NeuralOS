#pragma once

/// @file async_io_pool.h
/// @brief Thread-pool wrapper for async pread on macOS; thin io_uring wrapper on Linux.
///
/// On macOS, kqueue cannot async-complete file I/O, so a small pool of
/// dedicated I/O worker threads accepts pread requests via a task queue.
///
/// On Linux, this delegates to the existing IoUringBackend.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace nos {

class PlatformIO;

class AsyncIOPool {
public:
    explicit AsyncIOPool(int io_threads = 2, PlatformIO* platform_io = nullptr);
    ~AsyncIOPool();

    AsyncIOPool(const AsyncIOPool&) = delete;
    AsyncIOPool& operator=(const AsyncIOPool&) = delete;

    /// Submit an async read request.
    void submit_async_read(int fd, void* buf, size_t len, off_t offset,
                           std::function<void(int result)> on_complete);

    /// Process pending completions (Linux io_uring only; no-op on macOS).
    void drain_completions();

private:
    struct ReadRequest {
        int fd;
        void* buf;
        size_t len;
        off_t offset;
        std::function<void(int result)> on_complete;
    };

#ifdef __linux__
    PlatformIO* io_{nullptr};
    bool use_io_uring_{false};
#endif

    int io_threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<ReadRequest> request_queue_;
    std::atomic<int> in_flight_{0};
    std::atomic<bool> stop_{false};

    void io_worker_loop();
};

}  // namespace nos
