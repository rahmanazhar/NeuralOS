/// @file kmeans.cpp
/// @brief Balanced k-means clustering implementation.

#include "converter/kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nos {

namespace {

/// Simple xoshiro128** PRNG for reproducibility without stdlib dependency.
struct Rng {
    uint64_t state;

    explicit Rng(uint64_t seed) : state(seed == 0 ? 1 : seed) {}

    uint64_t next() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    /// Uniform double in [0, 1).
    double uniform() {
        return static_cast<double>(next() >> 11) / static_cast<double>(1ULL << 53);
    }
};

/// L2 squared distance between two vectors.
float l2_squared(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

/// K-means++ initialization: select k centroids with distance-proportional probability.
std::vector<std::vector<float>> kmeans_pp_init(const float* data, int n_points, int dim,
                                                int k, Rng& rng) {
    std::vector<std::vector<float>> centroids(static_cast<size_t>(k),
                                               std::vector<float>(static_cast<size_t>(dim)));

    // Pick first centroid uniformly at random
    int first = static_cast<int>(rng.next() % static_cast<uint64_t>(n_points));
    for (int d = 0; d < dim; d++) {
        centroids[0][static_cast<size_t>(d)] = data[first * dim + d];
    }

    std::vector<float> min_dist(static_cast<size_t>(n_points),
                                 std::numeric_limits<float>::max());

    for (int c = 1; c < k; c++) {
        // Update min distances to any chosen centroid
        for (int i = 0; i < n_points; i++) {
            float d = l2_squared(&data[i * dim], centroids[static_cast<size_t>(c) - 1].data(), dim);
            if (d < min_dist[static_cast<size_t>(i)]) {
                min_dist[static_cast<size_t>(i)] = d;
            }
        }

        // Compute cumulative distribution
        double total = 0.0;
        for (int i = 0; i < n_points; i++) {
            total += static_cast<double>(min_dist[static_cast<size_t>(i)]);
        }

        // Sample proportional to distance squared
        double threshold = rng.uniform() * total;
        double cumsum = 0.0;
        int chosen = 0;
        for (int i = 0; i < n_points; i++) {
            cumsum += static_cast<double>(min_dist[static_cast<size_t>(i)]);
            if (cumsum >= threshold) {
                chosen = i;
                break;
            }
        }

        for (int d = 0; d < dim; d++) {
            centroids[static_cast<size_t>(c)][static_cast<size_t>(d)] = data[chosen * dim + d];
        }
    }

    return centroids;
}

}  // namespace

KMeansResult balanced_kmeans(const float* data, int n_points, int dim, int k,
                              int max_iters, uint64_t seed) {
    Rng rng(seed);

    // Initialize centroids via k-means++
    auto centroids = kmeans_pp_init(data, n_points, dim, k, rng);

    // Balance constraint: STRICTLY equal cluster sizes when divisible.
    // The inference engine requires all experts have exactly intermediate_dim/expert_count rows,
    // so we enforce floor(n_points/k) max size. Remainder points (if any) get distributed
    // one per cluster starting from cluster 0.
    int base_size = n_points / k;
    int remainder = n_points % k;
    int max_size = base_size + (remainder > 0 ? 1 : 0);

    std::vector<int> assignments(static_cast<size_t>(n_points), -1);

    for (int iter = 0; iter < max_iters; iter++) {
        // --- Assignment step with balance constraint ---

        // Compute distance from each point to each centroid
        struct PointDist {
            int point;
            int cluster;
            float dist;
        };

        std::vector<PointDist> all_dists;
        all_dists.reserve(static_cast<size_t>(n_points) * static_cast<size_t>(k));

        for (int i = 0; i < n_points; i++) {
            for (int c = 0; c < k; c++) {
                float d = l2_squared(&data[i * dim], centroids[static_cast<size_t>(c)].data(), dim);
                all_dists.push_back({i, c, d});
            }
        }

        // Sort by distance (ascending) for greedy balanced assignment
        std::sort(all_dists.begin(), all_dists.end(),
                  [](const PointDist& a, const PointDist& b) { return a.dist < b.dist; });

        std::vector<int> new_assignments(static_cast<size_t>(n_points), -1);
        std::vector<int> cluster_sizes(static_cast<size_t>(k), 0);
        std::vector<bool> assigned(static_cast<size_t>(n_points), false);

        for (const auto& pd : all_dists) {
            if (assigned[static_cast<size_t>(pd.point)]) continue;
            if (cluster_sizes[static_cast<size_t>(pd.cluster)] >= max_size) continue;

            new_assignments[static_cast<size_t>(pd.point)] = pd.cluster;
            cluster_sizes[static_cast<size_t>(pd.cluster)]++;
            assigned[static_cast<size_t>(pd.point)] = true;
        }

        // Assign any remaining unassigned points to the nearest non-full cluster
        for (int i = 0; i < n_points; i++) {
            if (!assigned[static_cast<size_t>(i)]) {
                float best_dist = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < k; c++) {
                    if (cluster_sizes[static_cast<size_t>(c)] >= max_size) continue;
                    float d = l2_squared(&data[i * dim], centroids[static_cast<size_t>(c)].data(), dim);
                    if (d < best_dist) {
                        best_dist = d;
                        best_c = c;
                    }
                }
                new_assignments[static_cast<size_t>(i)] = best_c;
                cluster_sizes[static_cast<size_t>(best_c)]++;
            }
        }

        // Check convergence
        if (new_assignments == assignments) break;
        assignments = new_assignments;

        // --- Update step: recompute centroids ---
        for (int c = 0; c < k; c++) {
            std::fill(centroids[static_cast<size_t>(c)].begin(),
                      centroids[static_cast<size_t>(c)].end(), 0.0f);
        }

        for (int i = 0; i < n_points; i++) {
            int c = assignments[static_cast<size_t>(i)];
            for (int d = 0; d < dim; d++) {
                centroids[static_cast<size_t>(c)][static_cast<size_t>(d)] += data[i * dim + d];
            }
        }

        for (int c = 0; c < k; c++) {
            if (cluster_sizes[static_cast<size_t>(c)] > 0) {
                float inv = 1.0f / static_cast<float>(cluster_sizes[static_cast<size_t>(c)]);
                for (int d = 0; d < dim; d++) {
                    centroids[static_cast<size_t>(c)][static_cast<size_t>(d)] *= inv;
                }
            }
        }
    }

    // Build result
    KMeansResult result;
    result.clusters.resize(static_cast<size_t>(k));
    result.centroids = std::move(centroids);

    for (int i = 0; i < n_points; i++) {
        int c = assignments[static_cast<size_t>(i)];
        result.clusters[static_cast<size_t>(c)].push_back(static_cast<uint32_t>(i));
    }

    return result;
}

}  // namespace nos
