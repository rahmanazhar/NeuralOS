/// @file thread_pool.cpp
/// @brief Persistent thread pool implementation.
///
/// Workers sleep until tasks are available. dispatch_batch() enqueues tasks,
/// wakes workers, and blocks until all tasks complete (tracked by atomic counter).

#include "engine/thread_pool.h"

#include <cassert>

namespace nos {

ThreadPool::ThreadPool(int num_threads) : num_threads_(num_threads) {
    assert(num_threads > 0);

    workers_.reserve(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    cv_work_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::dispatch_batch(std::span<std::function<void()>> tasks) {
    if (tasks.empty()) return;

    // Set pending count before enqueuing so workers can decrement
    tasks_pending_.store(static_cast<int>(tasks.size()), std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& task : tasks) {
            task_queue_.push_back(std::move(task));
        }
    }
    cv_work_.notify_all();

    // Block until all tasks complete
    std::unique_lock<std::mutex> lock(done_mutex_);
    cv_done_.wait(lock, [this]() {
        return tasks_pending_.load(std::memory_order_acquire) <= 0;
    });
}

int ThreadPool::num_threads() const {
    return num_threads_;
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_work_.wait(lock, [this]() {
                return !task_queue_.empty() ||
                       stop_.load(std::memory_order_acquire);
            });

            if (stop_.load(std::memory_order_acquire) && task_queue_.empty()) {
                return;
            }

            if (task_queue_.empty()) continue;

            task = std::move(task_queue_.front());
            task_queue_.pop_front();
        }

        if (task) task();

        // Decrement pending and notify dispatcher if all done
        if (tasks_pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(done_mutex_);
            cv_done_.notify_one();
        }
    }
}

}  // namespace nos
