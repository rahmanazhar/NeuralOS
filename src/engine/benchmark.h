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

#include <string>
#include <vector>

namespace nos {

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

private:
    Config config_;
    std::string model_name_;
    int tokens_generated_ = 0;
    double total_time_ms_ = 0.0;
    double ttft_ms_ = 0.0;
};

}  // namespace nos
