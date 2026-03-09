/// @file benchmark.cpp
/// @brief Benchmark output: CSV, JSON, and LaTeX generation.

#include "engine/benchmark.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace nos {

BenchmarkReporter::BenchmarkReporter(Config config)
    : config_(std::move(config)) {}

void BenchmarkReporter::set_run_info(const std::string& model_name,
                                      int tokens_generated,
                                      double total_time_ms,
                                      double ttft_ms) {
    model_name_ = model_name;
    tokens_generated_ = tokens_generated;
    total_time_ms_ = total_time_ms;
    ttft_ms_ = ttft_ms;
}

void BenchmarkReporter::write(const MetricsCollector& metrics,
                               const VmmStats& vmm_stats,
                               const StickyRouter::AggregateMetrics& routing_metrics) const {
    std::filesystem::create_directories(config_.output_dir);

    write_csv(config_.output_dir + "/benchmark_results.csv",
              metrics, vmm_stats, routing_metrics);
    write_json(config_.output_dir + "/benchmark_results.json",
               metrics, vmm_stats, routing_metrics);
    write_latex(config_.output_dir + "/benchmark_tables.tex",
                metrics, vmm_stats, routing_metrics);
}

// ── CSV ─────────────────────────────────────────────────────────────────────

void BenchmarkReporter::write_csv(const std::string& path,
                                   const MetricsCollector& metrics,
                                   const VmmStats& vmm_stats,
                                   const StickyRouter::AggregateMetrics& routing_metrics) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    // Header
    ofs << "model,tokens,total_time_ms,tok_per_sec,ttft_ms,"
           "latency_p50_ms,latency_p95_ms,latency_p99_ms,"
           "cache_hit_rate,switch_rate,avg_stickiness_window,"
           "shift_detections,expert_reuse_ratio\n";

    auto lat = metrics.get_histogram("token_latency_ms");
    double tok_per_sec = (total_time_ms_ > 0)
        ? static_cast<double>(tokens_generated_) / (total_time_ms_ / 1000.0)
        : 0.0;

    double cache_hit_rate = (vmm_stats.total_pins > 0)
        ? static_cast<double>(vmm_stats.cache_hits) / static_cast<double>(vmm_stats.total_pins)
        : 0.0;

    uint64_t expert_reuses = metrics.get_counter("expert_reuses");
    uint64_t expert_loads = metrics.get_counter("expert_loads");
    double expert_reuse_ratio = (expert_reuses + expert_loads > 0)
        ? static_cast<double>(expert_reuses) / static_cast<double>(expert_reuses + expert_loads)
        : 0.0;

    uint64_t shift_detections = metrics.get_counter("shift_detections");

    // Data row
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "%s,%d,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.1f,%llu,%.4f\n",
        model_name_.c_str(),
        tokens_generated_,
        total_time_ms_,
        tok_per_sec,
        ttft_ms_,
        lat.p50,
        lat.p95,
        lat.p99,
        cache_hit_rate,
        static_cast<double>(routing_metrics.switch_rate),
        static_cast<double>(routing_metrics.avg_window_length),
        static_cast<unsigned long long>(shift_detections),
        expert_reuse_ratio);
    ofs << buf;
}

// ── JSON ────────────────────────────────────────────────────────────────────

void BenchmarkReporter::write_json(const std::string& path,
                                    const MetricsCollector& metrics,
                                    const VmmStats& vmm_stats,
                                    const StickyRouter::AggregateMetrics& routing_metrics) const {
    nlohmann::json j;

    // Run info
    double tok_per_sec = (total_time_ms_ > 0)
        ? static_cast<double>(tokens_generated_) / (total_time_ms_ / 1000.0)
        : 0.0;

    j["run_info"] = {
        {"model", model_name_},
        {"tokens_generated", tokens_generated_},
        {"total_time_ms", total_time_ms_},
        {"tok_per_sec", tok_per_sec},
        {"ttft_ms", ttft_ms_}
    };

    // VMM stats
    double cache_hit_rate = (vmm_stats.total_pins > 0)
        ? static_cast<double>(vmm_stats.cache_hits) / static_cast<double>(vmm_stats.total_pins)
        : 0.0;

    j["vmm_stats"] = {
        {"total_pins", vmm_stats.total_pins},
        {"cache_hits", vmm_stats.cache_hits},
        {"cache_misses", vmm_stats.cache_misses},
        {"evictions", vmm_stats.evictions},
        {"cache_hit_rate", cache_hit_rate}
    };

    // Routing metrics
    j["routing"] = {
        {"total_decisions", routing_metrics.total_routing_decisions},
        {"total_switches", routing_metrics.total_switches},
        {"switch_rate", routing_metrics.switch_rate},
        {"avg_window_length", routing_metrics.avg_window_length}
    };

    // All metrics data
    j["metrics"] = metrics.to_json();

    std::ofstream ofs(path);
    if (!ofs.is_open()) return;
    ofs << j.dump(2) << "\n";
}

// ── LaTeX ───────────────────────────────────────────────────────────────────

std::string BenchmarkReporter::escape_latex(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&':  out += "\\&"; break;
            case '%':  out += "\\%"; break;
            case '$':  out += "\\$"; break;
            case '#':  out += "\\#"; break;
            case '_':  out += "\\_"; break;
            case '{':  out += "\\{"; break;
            case '}':  out += "\\}"; break;
            case '~':  out += "\\textasciitilde{}"; break;
            case '^':  out += "\\textasciicircum{}"; break;
            default:   out += c; break;
        }
    }
    return out;
}

std::string BenchmarkReporter::generate_latex_table(
        const std::string& caption,
        const std::vector<std::string>& headers,
        const std::vector<std::vector<std::string>>& rows) const {
    std::ostringstream ss;

    ss << "\\begin{table}[htbp]\n";
    ss << "  \\centering\n";
    ss << "  \\caption{" << escape_latex(caption) << "}\n";
    ss << "  \\begin{tabular}{";
    for (size_t i = 0; i < headers.size(); i++) {
        ss << (i == 0 ? "l" : "r");
    }
    ss << "}\n";
    ss << "    \\toprule\n";

    // Header row
    ss << "    ";
    for (size_t i = 0; i < headers.size(); i++) {
        if (i > 0) ss << " & ";
        ss << escape_latex(headers[i]);
    }
    ss << " \\\\\n";
    ss << "    \\midrule\n";

    // Data rows
    for (const auto& row : rows) {
        ss << "    ";
        for (size_t i = 0; i < row.size(); i++) {
            if (i > 0) ss << " & ";
            ss << row[i];  // Not escaped — values are pre-formatted
        }
        ss << " \\\\\n";
    }

    ss << "    \\bottomrule\n";
    ss << "  \\end{tabular}\n";
    ss << "\\end{table}\n";

    return ss.str();
}

void BenchmarkReporter::write_latex(const std::string& path,
                                     const MetricsCollector& metrics,
                                     const VmmStats& vmm_stats,
                                     const StickyRouter::AggregateMetrics& routing_metrics) const {
    auto lat = metrics.get_histogram("token_latency_ms");
    double tok_per_sec = (total_time_ms_ > 0)
        ? static_cast<double>(tokens_generated_) / (total_time_ms_ / 1000.0)
        : 0.0;

    double cache_hit_rate = (vmm_stats.total_pins > 0)
        ? static_cast<double>(vmm_stats.cache_hits) / static_cast<double>(vmm_stats.total_pins)
        : 0.0;

    uint64_t expert_reuses = metrics.get_counter("expert_reuses");
    uint64_t expert_loads = metrics.get_counter("expert_loads");
    double expert_reuse_ratio = (expert_reuses + expert_loads > 0)
        ? static_cast<double>(expert_reuses) / static_cast<double>(expert_reuses + expert_loads)
        : 0.0;

    std::ostringstream ss;

    // Header comment
    ss << "% NeuralOS Benchmark Tables\n";
    ss << "% Generated by: neuralos run --bench\n";
    ss << "% Usage: \\input{benchmark_tables.tex} in your paper\n";
    ss << "% Model: " << escape_latex(model_name_) << "\n\n";

    if (config_.standalone_latex) {
        ss << "\\documentclass{article}\n";
        ss << "\\usepackage{booktabs}\n";
        ss << "\\begin{document}\n\n";
    }

    // Table 1: Performance Summary
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f", tok_per_sec);
    std::string tok_s_str(buf);
    std::snprintf(buf, sizeof(buf), "%.1f", ttft_ms_);
    std::string ttft_str(buf);
    std::snprintf(buf, sizeof(buf), "%.2f", lat.p50);
    std::string p50_str(buf);
    std::snprintf(buf, sizeof(buf), "%.2f", lat.p95);
    std::string p95_str(buf);
    std::snprintf(buf, sizeof(buf), "%.2f", lat.p99);
    std::string p99_str(buf);

    ss << generate_latex_table("Performance Summary",
        {"Metric", "Value"},
        {
            {"tok/s", tok_s_str},
            {"TTFT (ms)", ttft_str},
            {"Latency p50 (ms)", p50_str},
            {"Latency p95 (ms)", p95_str},
            {"Latency p99 (ms)", p99_str},
        });

    ss << "\n";

    // Table 2: Routing Analysis
    std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(routing_metrics.switch_rate));
    std::string sr_str(buf);
    std::snprintf(buf, sizeof(buf), "%.1f", static_cast<double>(routing_metrics.avg_window_length));
    std::string awl_str(buf);
    std::string sd_str = std::to_string(metrics.get_counter("shift_detections"));

    ss << generate_latex_table("Routing Analysis",
        {"Metric", "Value"},
        {
            {"Switch Rate", sr_str},
            {"Avg Stickiness Window", awl_str},
            {"Shift Detections", sd_str},
        });

    ss << "\n";

    // Table 3: Memory Efficiency
    std::snprintf(buf, sizeof(buf), "%.4f", cache_hit_rate);
    std::string chr_str(buf);
    std::snprintf(buf, sizeof(buf), "%.4f", expert_reuse_ratio);
    std::string err_str(buf);
    std::string ev_str = std::to_string(vmm_stats.evictions);

    ss << generate_latex_table("Memory Efficiency",
        {"Metric", "Value"},
        {
            {"Cache Hit Rate", chr_str},
            {"Expert Reuse Ratio", err_str},
            {"Evictions", ev_str},
        });

    if (config_.standalone_latex) {
        ss << "\n\\end{document}\n";
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) return;
    ofs << ss.str();
}

}  // namespace nos
