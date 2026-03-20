/// @file dashboard.cpp
/// @brief ncurses TUI dashboard implementation.
///
/// All ncurses calls happen on the main thread (ncurses is NOT thread-safe).
/// MetricsReader reads from shared memory; display updates are single-threaded.

#include "dashboard/dashboard.h"
#include "dashboard/sparkline.h"
#include "server/shared_metrics.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>

#include <curses.h>

namespace nos {

// ── Color pairs ─────────────────────────────────────────────────────────────
static constexpr short COLOR_GOOD    = 1;
static constexpr short COLOR_MODERATE = 2;
static constexpr short COLOR_POOR    = 3;
static constexpr short COLOR_TITLE   = 4;
static constexpr short COLOR_WAITING = 5;
static constexpr short COLOR_PAUSED  = 6;

// ── Minimum terminal dimensions ─────────────────────────────────────────────
static constexpr int MIN_COLS = 80;
static constexpr int MIN_ROWS = 24;

struct Dashboard::Impl {
    Config config;
    WINDOW* panels[4] = {};  // TL, TR, BL, BR
    int focused_panel = 0;
    bool paused = false;
    bool initialized = false;

    void draw_panel_border(int idx, const char* title);
    void draw_performance(const SharedMetrics& m, int half_y, int half_x);
    void draw_cache(const SharedMetrics& m, int half_y, int half_x);
    void draw_prefetch(const SharedMetrics& m, int half_y, int half_x);
    void draw_routing(const SharedMetrics& m, int half_y, int half_x);
    void save_snapshot(const SharedMetrics& m);
};

Dashboard::Dashboard() : impl_(new Impl()) {}

Dashboard::~Dashboard() {
    stop();
    delete impl_;
}

bool Dashboard::start(const Config& config) {
    impl_->config = config;

    // Initialize ncurses
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);

    // Initialize colors
    if (has_colors()) {
        start_color();
        use_default_colors();
        init_pair(COLOR_GOOD, COLOR_GREEN, -1);
        init_pair(COLOR_MODERATE, COLOR_YELLOW, -1);
        init_pair(COLOR_POOR, COLOR_RED, -1);
        init_pair(COLOR_TITLE, COLOR_CYAN, -1);
        init_pair(COLOR_WAITING, COLOR_YELLOW, -1);
        init_pair(COLOR_PAUSED, COLOR_RED, -1);
    }

    // Check terminal size
    int max_y = 0;
    int max_x = 0;
    getmaxyx(stdscr, max_y, max_x);
    if (max_x < MIN_COLS || max_y < MIN_ROWS) {
        endwin();
        std::fprintf(stderr,
            "Terminal too small: %dx%d (need at least %dx%d)\n",
            max_x, max_y, MIN_COLS, MIN_ROWS);
        return false;
    }

    // Create 2x2 panel grid
    int half_y = max_y / 2;
    int half_x = max_x / 2;

    impl_->panels[0] = newwin(half_y, half_x, 0, 0);          // Top-left: Performance
    impl_->panels[1] = newwin(half_y, max_x - half_x, 0, half_x); // Top-right: Cache
    impl_->panels[2] = newwin(max_y - half_y, half_x, half_y, 0); // Bottom-left: Prefetch
    impl_->panels[3] = newwin(max_y - half_y, max_x - half_x, half_y, half_x); // Bottom-right: Routing

    impl_->initialized = true;
    return true;
}

void Dashboard::run() {
    if (!impl_->initialized) return;

    MetricsReader reader(impl_->config.shm_name);

    while (true) {
        // Handle input (non-blocking)
        int ch = getch();
        if (ch == 'q' || ch == 'Q' || ch == 27) {  // 27 = ESC
            break;
        } else if (ch == '\t' || ch == KEY_BTAB) {
            impl_->focused_panel = (impl_->focused_panel + 1) % 4;
        } else if (ch == 's' || ch == 'S') {
            SharedMetrics m = reader.read();
            impl_->save_snapshot(m);
        } else if (ch == 'p' || ch == 'P') {
            impl_->paused = !impl_->paused;
        }

        // Get terminal dimensions for rendering
        int max_y = 0;
        int max_x = 0;
        getmaxyx(stdscr, max_y, max_x);
        int half_y = max_y / 2;
        int half_x = max_x / 2;

        if (!impl_->paused) {
            if (!reader.is_open() || !reader.is_valid()) {
                // Show waiting message
                for (int i = 0; i < 4; ++i) {
                    werase(impl_->panels[i]);
                    box(impl_->panels[i], 0, 0);
                }
                const char* titles[] = {" Performance ", " Cache ", " Prefetch ", " Routing "};
                for (int i = 0; i < 4; ++i) {
                    impl_->draw_panel_border(i, titles[i]);
                }

                wattron(impl_->panels[0], COLOR_PAIR(COLOR_WAITING));
                mvwprintw(impl_->panels[0], half_y / 2, 3, "Waiting for server...");
                mvwprintw(impl_->panels[0], half_y / 2 + 1, 3,
                          "Connect to: %s", impl_->config.shm_name.c_str());
                wattroff(impl_->panels[0], COLOR_PAIR(COLOR_WAITING));

                for (int i = 0; i < 4; ++i) {
                    wrefresh(impl_->panels[i]);
                }
            } else {
                SharedMetrics m = reader.read();

                impl_->draw_performance(m, half_y, half_x);
                impl_->draw_cache(m, half_y, half_x);
                impl_->draw_prefetch(m, half_y, half_x);
                impl_->draw_routing(m, half_y, half_x);
            }
        } else {
            // Show PAUSED indicator
            wattron(impl_->panels[0], COLOR_PAIR(COLOR_PAUSED) | A_BOLD);
            mvwprintw(impl_->panels[0], 1, 3, " PAUSED ");
            wattroff(impl_->panels[0], COLOR_PAIR(COLOR_PAUSED) | A_BOLD);
            for (int i = 0; i < 4; ++i) {
                wrefresh(impl_->panels[i]);
            }
        }

        napms(impl_->config.refresh_ms);
    }
}

void Dashboard::stop() {
    if (!impl_->initialized) return;

    for (int i = 0; i < 4; ++i) {
        if (impl_->panels[i] != nullptr) {
            delwin(impl_->panels[i]);
            impl_->panels[i] = nullptr;
        }
    }

    endwin();
    impl_->initialized = false;
}

// ── Panel rendering helpers ─────────────────────────────────────────────────

void Dashboard::Impl::draw_panel_border(int idx, const char* title) {
    WINDOW* win = panels[idx];
    if (idx == focused_panel) {
        wattron(win, A_BOLD);
        box(win, 0, 0);
        wattroff(win, A_BOLD);
    } else {
        box(win, 0, 0);
    }
    wattron(win, COLOR_PAIR(COLOR_TITLE) | A_BOLD);
    mvwprintw(win, 0, 2, "%s", title);
    wattroff(win, COLOR_PAIR(COLOR_TITLE) | A_BOLD);
}

void Dashboard::Impl::draw_performance(const SharedMetrics& m,
                                        int /*half_y*/, int half_x) {
    WINDOW* win = panels[0];
    werase(win);
    draw_panel_border(0, " Performance ");

    int row = 2;

    // Tok/s with color coding
    short color = COLOR_GOOD;
    if (m.tok_per_sec < 5.0) color = COLOR_POOR;
    else if (m.tok_per_sec < 20.0) color = COLOR_MODERATE;

    wattron(win, COLOR_PAIR(color));
    mvwprintw(win, row++, 3, "tok/s:      %.1f", m.tok_per_sec);
    wattroff(win, COLOR_PAIR(color));

    mvwprintw(win, row++, 3, "TTFT:       %.1f ms", m.ttft_ms);
    mvwprintw(win, row++, 3, "Latency p50: %.1f ms", m.latency_p50_ms);
    mvwprintw(win, row++, 3, "Latency p95: %.1f ms", m.latency_p95_ms);
    mvwprintw(win, row++, 3, "Latency p99: %.1f ms", m.latency_p99_ms);

    // Slots
    row++;
    mvwprintw(win, row++, 3, "Slots:      %u/%u", m.active_slots, m.max_slots);

    // Sparkline
    row++;
    // Determine sparkline width (panel width minus borders and label)
    int spark_width = half_x - 8;
    if (spark_width < 10) spark_width = 10;

    // Extract history from circular buffer
    size_t usable = static_cast<size_t>(spark_width);
    if (usable > 120) usable = 120;

    // Read from circular buffer starting at oldest entry
    float ordered[120];
    for (size_t i = 0; i < usable; ++i) {
        uint32_t idx = (m.history_write_idx + static_cast<uint32_t>(120 - usable + i)) % 120;
        ordered[i] = m.tok_per_sec_history[idx];
    }

    std::string sparkline = render_sparkline(ordered, usable, usable);
    mvwprintw(win, row, 3, "%s", sparkline.c_str());

    wrefresh(win);
}

void Dashboard::Impl::draw_cache(const SharedMetrics& m,
                                  int /*half_y*/, int /*half_x*/) {
    WINDOW* win = panels[1];
    werase(win);
    draw_panel_border(1, " Cache ");

    int row = 2;

    // Hit rate with color coding
    short color = COLOR_GOOD;
    if (m.cache_hit_rate < 0.70) color = COLOR_POOR;
    else if (m.cache_hit_rate < 0.85) color = COLOR_MODERATE;

    wattron(win, COLOR_PAIR(color));
    mvwprintw(win, row++, 3, "Hit Rate:   %.1f%%",
              m.cache_hit_rate * 100.0);
    wattroff(win, COLOR_PAIR(color));

    mvwprintw(win, row++, 3, "Evictions:  %llu",
              static_cast<unsigned long long>(m.evictions));
    mvwprintw(win, row++, 3, "Resident:   %u experts", m.resident_experts);

    wrefresh(win);
}

void Dashboard::Impl::draw_prefetch(const SharedMetrics& m,
                                     int /*half_y*/, int /*half_x*/) {
    WINDOW* win = panels[2];
    werase(win);
    draw_panel_border(2, " Prefetch ");

    int row = 2;

    // Oracle RWP with color coding
    short color = COLOR_GOOD;
    if (m.oracle_rwp < 0.50) color = COLOR_POOR;
    else if (m.oracle_rwp < 0.80) color = COLOR_MODERATE;

    wattron(win, COLOR_PAIR(color));
    mvwprintw(win, row++, 3, "Oracle RWP: %.1f%%",
              m.oracle_rwp * 100.0);
    wattroff(win, COLOR_PAIR(color));

    mvwprintw(win, row++, 3, "Waste:      %.1f%%",
              m.waste_ratio * 100.0);
    mvwprintw(win, row++, 3, "Mode:       %s", m.prefetch_mode);

    wrefresh(win);
}

void Dashboard::Impl::draw_routing(const SharedMetrics& m,
                                    int /*half_y*/, int /*half_x*/) {
    WINDOW* win = panels[3];
    werase(win);
    draw_panel_border(3, " Routing ");

    int row = 2;

    mvwprintw(win, row++, 3, "Switch Rate: %.1f%%",
              m.switch_rate * 100.0);

    // Sticky pct with color coding (higher is better for cache locality)
    short color = COLOR_GOOD;
    if (m.sticky_pct < 0.50) color = COLOR_POOR;
    else if (m.sticky_pct < 0.75) color = COLOR_MODERATE;

    wattron(win, COLOR_PAIR(color));
    mvwprintw(win, row++, 3, "Sticky %%:   %.1f%%",
              m.sticky_pct * 100.0);
    wattroff(win, COLOR_PAIR(color));

    mvwprintw(win, row++, 3, "Shifts:     %u", m.shift_detections);

    wrefresh(win);
}

void Dashboard::Impl::save_snapshot(const SharedMetrics& m) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    char time_buf[64];
    std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S",
                  std::localtime(&time_t_now));

    std::string filename = "neuralos_dashboard_snapshot_";
    filename += time_buf;
    filename += ".txt";

    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;

    ofs << "NeuralOS Dashboard Snapshot -- " << time_buf << "\n";
    ofs << "============================================\n\n";

    ofs << "[Performance]\n";
    ofs << "  tok/s:        " << m.tok_per_sec << "\n";
    ofs << "  TTFT:         " << m.ttft_ms << " ms\n";
    ofs << "  Latency p50:  " << m.latency_p50_ms << " ms\n";
    ofs << "  Latency p95:  " << m.latency_p95_ms << " ms\n";
    ofs << "  Latency p99:  " << m.latency_p99_ms << " ms\n";
    ofs << "  Active Slots: " << m.active_slots << "/" << m.max_slots << "\n\n";

    ofs << "[Cache]\n";
    ofs << "  Hit Rate:     " << (m.cache_hit_rate * 100.0) << "%\n";
    ofs << "  Evictions:    " << m.evictions << "\n";
    ofs << "  Resident:     " << m.resident_experts << " experts\n\n";

    ofs << "[Prefetch]\n";
    ofs << "  Oracle RWP:   " << (m.oracle_rwp * 100.0) << "%\n";
    ofs << "  Waste Ratio:  " << (m.waste_ratio * 100.0) << "%\n";
    ofs << "  Mode:         " << m.prefetch_mode << "\n\n";

    ofs << "[Routing]\n";
    ofs << "  Switch Rate:  " << (m.switch_rate * 100.0) << "%\n";
    ofs << "  Sticky %%:     " << (m.sticky_pct * 100.0) << "%\n";
    ofs << "  Shift Detections: " << m.shift_detections << "\n";

    // Flash message on screen
    wattron(panels[0], A_BOLD | COLOR_PAIR(COLOR_GOOD));
    mvwprintw(panels[0], 1, 3, " Saved: %s ", filename.c_str());
    wattroff(panels[0], A_BOLD | COLOR_PAIR(COLOR_GOOD));
    wrefresh(panels[0]);
}

}  // namespace nos
