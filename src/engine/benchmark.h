#pragma once

/// @file benchmark.h
/// @brief Benchmark output: CSV, JSON, and LaTeX table generation.
///
/// Reads MetricsCollector snapshots and VmmStats/routing metrics to produce
/// paper-ready output files. LaTeX uses booktabs formatting with optional
/// standalone document wrapping.

#include "engine/metrics.h"
#include "engine/oracle_prefetcher.h"
#include "engine/sticky_router.h"
#include "vmm/vmm.h"

#include <cstddef>
#include <string>
#include <vector>

namespace nos {

/// Captures one benchmark run's data for paper Table 1 generation.
struct BenchRunResult {
    std::string model_name;
    std::string model_size;           ///< e.g., "70B"
    int tokens_generated = 0;
    double total_time_ms = 0.0;
    double ttft_ms = 0.0;
    double tok_per_sec = 0.0;
    double cache_hit_rate = 0.0;
    double switch_rate = 0.0;
    double avg_sticky_window = 0.0;
    double prefetch_rwp = 0.0;
    std::string prefetch_mode = "none";
    int effective_k = 0;
    double waste_ratio = 0.0;
    size_t memory_budget_mb = 0;
    int num_threads = 0;
    int concurrent_sequences = 0;     ///< for multi-seq batch test
    double multi_seq_tok_per_sec = 0.0; ///< aggregate throughput with batching
};

class BenchmarkReporter {
public:
    struct Config {
        std::string output_dir = "benchmark_results";
        bool standalone_latex = false;
    };

    explicit BenchmarkReporter(Config config);

    /// Set overall run metadata.
    void set_run_info(const std::string& model_name, int tokens_generated,
                      double total_time_ms, double ttft_ms);

    /// Write all output files (CSV, JSON, LaTeX) to output_dir.
    void write(const MetricsCollector& metrics,
               const VmmStats& vmm_stats,
               const StickyRouter::AggregateMetrics& routing_metrics) const;

    /// Overload with optional prefetch stats.
    void write(const MetricsCollector& metrics,
               const VmmStats& vmm_stats,
               const StickyRouter::AggregateMetrics& routing_metrics,
               const PrefetchStats& prefetch_stats) const;

    /// Individual writers (public for testing).
    void write_csv(const std::string& path, const MetricsCollector& metrics,
                   const VmmStats& vmm_stats,
                   const StickyRouter::AggregateMetrics& routing_metrics) const;

    void write_csv(const std::string& path, const MetricsCollector& metrics,
                   const VmmStats& vmm_stats,
                   const StickyRouter::AggregateMetrics& routing_metrics,
                   const PrefetchStats& prefetch_stats) const;

    void write_json(const std::string& path, const MetricsCollector& metrics,
                    const VmmStats& vmm_stats,
                    const StickyRouter::AggregateMetrics& routing_metrics) const;

    void write_json(const std::string& path, const MetricsCollector& metrics,
                    const VmmStats& vmm_stats,
                    const StickyRouter::AggregateMetrics& routing_metrics,
                    const PrefetchStats& prefetch_stats) const;

    void write_latex(const std::string& path, const MetricsCollector& metrics,
                     const VmmStats& vmm_stats,
                     const StickyRouter::AggregateMetrics& routing_metrics) const;

    /// LaTeX helpers (public for testing).
    static std::string escape_latex(const std::string& s);
    std::string generate_latex_table(
        const std::string& caption,
        const std::vector<std::string>& headers,
        const std::vector<std::vector<std::string>>& rows) const;

    // ── Paper Table 1 generation from BenchRunResult ────────────────────

    /// Write LaTeX table matching paper Table 1 format with booktabs.
    void write_paper_table(const std::string& path,
                           const std::vector<BenchRunResult>& results) const;

    /// Write comparison LaTeX table for multiple configurations.
    void write_comparison_table(const std::string& path,
                                const std::vector<BenchRunResult>& results) const;

    /// Write JSON array of all run results.
    void write_paper_json(const std::string& path,
                          const std::vector<BenchRunResult>& results) const;

    /// Write CSV with all fields for spreadsheet import.
    void write_paper_csv(const std::string& path,
                         const std::vector<BenchRunResult>& results) const;

private:
    Config config_;
    std::string model_name_;
    int tokens_generated_ = 0;
    double total_time_ms_ = 0.0;
    double ttft_ms_ = 0.0;
};

}  // namespace nos
