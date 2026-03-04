/// @file test_thread_pool.cpp
/// @brief ThreadPool tests: batch dispatch, barrier sync, multi-threaded execution.

#include <catch2/catch_test_macros.hpp>

#include "engine/thread_pool.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

TEST_CASE("ThreadPool: dispatch_batch with 1 task completes", "[thread_pool]") {
    nos::ThreadPool pool(2);

    std::atomic<int> counter{0};
    std::vector<std::function<void()>> tasks;
    tasks.push_back([&counter]() { counter.fetch_add(1); });

    auto span = std::span<std::function<void()>>(tasks);
    pool.dispatch_batch(span);

    CHECK(counter.load() == 1);
}

TEST_CASE("ThreadPool: dispatch_batch with 2 tasks completes all", "[thread_pool]") {
    nos::ThreadPool pool(2);

    std::atomic<int> counter{0};
    std::vector<std::function<void()>> tasks;
    tasks.push_back([&counter]() { counter.fetch_add(1); });
    tasks.push_back([&counter]() { counter.fetch_add(1); });

    auto span = std::span<std::function<void()>>(tasks);
    pool.dispatch_batch(span);

    CHECK(counter.load() == 2);
}

TEST_CASE("ThreadPool: dispatch_batch with 4 tasks completes all", "[thread_pool]") {
    nos::ThreadPool pool(4);

    std::atomic<int> counter{0};
    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 4; ++i) {
        tasks.push_back([&counter]() { counter.fetch_add(1); });
    }

    auto span = std::span<std::function<void()>>(tasks);
    pool.dispatch_batch(span);

    CHECK(counter.load() == 4);
}

TEST_CASE("ThreadPool: tasks execute on different threads", "[thread_pool]") {
    nos::ThreadPool pool(4);

    std::mutex mtx;
    std::set<std::thread::id> thread_ids;

    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 4; ++i) {
        tasks.push_back([&mtx, &thread_ids]() {
            // Small sleep so multiple workers pick up tasks
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            auto id = std::this_thread::get_id();
            std::lock_guard<std::mutex> lock(mtx);
            thread_ids.insert(id);
        });
    }

    auto span = std::span<std::function<void()>>(tasks);
    pool.dispatch_batch(span);

    // At least 2 different threads should have executed tasks
    // (on a multi-core machine, likely all 4)
    CHECK(thread_ids.size() >= 2);
}

TEST_CASE("ThreadPool: empty batch completes immediately", "[thread_pool]") {
    nos::ThreadPool pool(2);

    std::vector<std::function<void()>> tasks;
    auto span = std::span<std::function<void()>>(tasks);

    auto start = std::chrono::steady_clock::now();
    pool.dispatch_batch(span);
    auto end = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    CHECK(elapsed.count() < 10);  // Should be essentially instant
}

TEST_CASE("ThreadPool: sequential batch dispatches reuse pool", "[thread_pool]") {
    nos::ThreadPool pool(2);

    for (int batch = 0; batch < 5; ++batch) {
        std::atomic<int> counter{0};
        std::vector<std::function<void()>> tasks;
        tasks.push_back([&counter]() { counter.fetch_add(1); });
        tasks.push_back([&counter]() { counter.fetch_add(1); });

        auto span = std::span<std::function<void()>>(tasks);
        pool.dispatch_batch(span);

        CHECK(counter.load() == 2);
    }
}

TEST_CASE("ThreadPool: num_threads returns correct count", "[thread_pool]") {
    {
        nos::ThreadPool pool(1);
        CHECK(pool.num_threads() == 1);
    }
    {
        nos::ThreadPool pool(4);
        CHECK(pool.num_threads() == 4);
    }
    {
        nos::ThreadPool pool(8);
        CHECK(pool.num_threads() == 8);
    }
}

TEST_CASE("ThreadPool: destructor joins cleanly without deadlock", "[thread_pool]") {
    // This test verifies RAII cleanup works.
    // If the destructor deadlocks, this test will timeout.
    auto start = std::chrono::steady_clock::now();
    {
        nos::ThreadPool pool(4);

        std::atomic<int> counter{0};
        std::vector<std::function<void()>> tasks;
        tasks.push_back([&counter]() { counter.fetch_add(1); });
        tasks.push_back([&counter]() { counter.fetch_add(1); });

        auto span = std::span<std::function<void()>>(tasks);
        pool.dispatch_batch(span);
    }
    // Pool destroyed here -- if we reach this line, no deadlock
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    CHECK(elapsed.count() < 5);  // Should complete almost instantly
}

TEST_CASE("ThreadPool: more tasks than threads are all completed", "[thread_pool]") {
    nos::ThreadPool pool(2);

    std::atomic<int> counter{0};
    std::vector<std::function<void()>> tasks;
    for (int i = 0; i < 8; ++i) {
        tasks.push_back([&counter]() { counter.fetch_add(1); });
    }

    auto span = std::span<std::function<void()>>(tasks);
    pool.dispatch_batch(span);

    CHECK(counter.load() == 8);
}
