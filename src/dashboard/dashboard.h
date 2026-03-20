#pragma once

/// @file dashboard.h
/// @brief ncurses TUI dashboard controller for live inference metrics.
///
/// Displays four panels (Performance, Cache, Prefetch, Routing) with
/// real-time data from the server's SharedMetrics via POSIX shared memory.
/// Refreshes at 2 Hz, shows sparkline history, supports tab navigation.

#include <string>

namespace nos {

class Dashboard {
public:
    struct Config {
        std::string shm_name;       ///< POSIX shm name (e.g., "/neuralos_metrics_12345")
        int refresh_ms = 500;       ///< Refresh interval in ms (500 = 2 Hz)
    };

    Dashboard();
    ~Dashboard();

    Dashboard(const Dashboard&) = delete;
    Dashboard& operator=(const Dashboard&) = delete;

    /// Initialize ncurses, create panels, and prepare for rendering.
    /// @return true if initialization succeeded
    bool start(const Config& config);

    /// Main event loop (blocking). Handles input + rendering.
    /// Returns when user presses 'q' or ESC.
    void run();

    /// Cleanup ncurses state.
    void stop();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace nos
