#pragma once

/// @file thread_pool.h
/// @brief Persistent thread pool with batch dispatch.
///
/// Workers are std::thread instances that sleep on a condition_variable
/// between batches. Dispatch blocks until all tasks complete, tracked
/// by an atomic counter with condvar notification.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace nos {

class ThreadPool {
public:
    explicit ThreadPool(int num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Dispatch a batch of tasks and block until all complete.
    void dispatch_batch(std::span<std::function<void()>> tasks);

    int num_threads() const;

private:
    void worker_loop();

    int num_threads_;
    std::vector<std::thread> workers_;

    // Task queue
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::deque<std::function<void()>> task_queue_;

    // Completion signaling: dispatcher waits for tasks_pending_ == 0
    std::atomic<int> tasks_pending_{0};
    std::mutex done_mutex_;
    std::condition_variable cv_done_;

    std::atomic<bool> stop_{false};
};

}  // namespace nos
