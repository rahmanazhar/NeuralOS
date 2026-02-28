#pragma once

/// @file rmsnorm.h
/// @brief RMSNorm layer normalization.

#include <cstddef>

namespace nos {

/// Apply RMS Layer Normalization.
///
/// out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
///
/// @param out     Output buffer (dim floats)
/// @param x       Input buffer (dim floats)
/// @param weight  Learnable scale weights (dim floats)
/// @param dim     Dimension of vectors
/// @param eps     Epsilon for numerical stability
void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps);

}  // namespace nos
