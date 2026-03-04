/// @file async_io_pool.cpp
/// @brief AsyncIOPool implementation.
///
/// On macOS: small pool of std::thread workers executing pread requests.
/// On Linux with io_uring: thin wrapper delegating to PlatformIO.

#include "io/async_io_pool.h"
#include "io/platform_io.h"

#include <cerrno>
#include <unistd.h>

namespace nos {

AsyncIOPool::AsyncIOPool(int io_threads, PlatformIO* platform_io)
    : io_threads_(io_threads) {

#ifdef __linux__
    io_ = platform_io;
    #ifdef NOS_HAS_IO_URING
    if (io_) {
        use_io_uring_ = true;
        return;  // No worker threads needed
    }
    #endif
#else
    (void)platform_io;
#endif

    // macOS path (or Linux fallback): launch I/O worker threads
    workers_.reserve(static_cast<size_t>(io_threads_));
    for (int i = 0; i < io_threads_; ++i) {
        workers_.emplace_back(&AsyncIOPool::io_worker_loop, this);
    }
}

AsyncIOPool::~AsyncIOPool() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void AsyncIOPool::submit_async_read(int fd, void* buf, size_t len, off_t offset,
                                     std::function<void(int result)> on_complete) {
#ifdef __linux__
    if (use_io_uring_ && io_) {
        struct CallbackCtx {
            std::function<void(int)> cb;
        };
        auto* ctx = new CallbackCtx{std::move(on_complete)};
        io_->submit_read(fd, buf, len, offset, ctx);
        in_flight_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
#endif

    // macOS path: enqueue to worker thread pool
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request_queue_.push_back(ReadRequest{
            fd, buf, len, offset, std::move(on_complete)
        });
        in_flight_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_one();
}

void AsyncIOPool::drain_completions() {
#ifdef __linux__
    if (use_io_uring_ && io_) {
        IoCompletion completions[16];
        int n = io_->poll(completions, 16, 0);
        for (int i = 0; i < n; ++i) {
            if (completions[i].userdata) {
                struct CallbackCtx {
                    std::function<void(int)> cb;
                };
                auto* ctx = static_cast<CallbackCtx*>(completions[i].userdata);
                if (ctx->cb) {
                    ctx->cb(completions[i].result);
                }
                delete ctx;
                in_flight_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        return;
    }
#endif

    // macOS: callbacks fire on worker threads, nothing to drain
}

void AsyncIOPool::io_worker_loop() {
    while (!stop_.load(std::memory_order_acquire)) {
        ReadRequest req;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return !request_queue_.empty() ||
                       stop_.load(std::memory_order_acquire);
            });

            if (stop_.load(std::memory_order_acquire)) break;
            if (request_queue_.empty()) continue;

            req = std::move(request_queue_.front());
            request_queue_.pop_front();
        }

        // Execute pread synchronously on this I/O worker thread
        ssize_t bytes = ::pread(req.fd, req.buf, req.len, req.offset);
        int result = (bytes < 0) ? -errno : static_cast<int>(bytes);

        if (req.on_complete) {
            req.on_complete(result);
        }
        in_flight_.fetch_sub(1, std::memory_order_relaxed);
    }
}

}  // namespace nos
