#pragma once

/// @file kmeans.h
/// @brief Balanced k-means clustering for expert decomposition.
///
/// Clusters FFN neurons into expert groups based on activation magnitude
/// similarity, with balance constraints to prevent degenerate expert sizes.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nos {

/// Result of balanced k-means clustering.
struct KMeansResult {
    /// Per-cluster lists of point indices.
    std::vector<std::vector<uint32_t>> clusters;

    /// Per-cluster centroid vectors [k][dim].
    std::vector<std::vector<float>> centroids;
};

/// Balanced k-means clustering.
///
/// @param data       Input data matrix [n_points x dim], row-major
/// @param n_points   Number of data points (e.g., FFN neurons)
/// @param dim        Dimensionality of each point
/// @param k          Number of clusters (experts)
/// @param max_iters  Maximum iterations
/// @param seed       Random seed for k-means++ initialization
/// @return Clustering result with balanced cluster assignments
KMeansResult balanced_kmeans(const float* data, int n_points, int dim, int k,
                              int max_iters = 100, uint64_t seed = 42);

}  // namespace nos
