#include "engine/rmsnorm.h"
#include <cmath>

namespace nos {

void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps) {
    // Compute sum of squares
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    // Reciprocal RMS
    float rms = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
    // Scale and apply weight
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

}  // namespace nos
