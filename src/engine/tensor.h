#pragma once

/// @file tensor.h
/// @brief Lightweight non-owning tensor view for passing activation buffers.

#include <cassert>
#include <cstddef>

namespace nos {

/// Non-owning tensor view — wraps caller-allocated buffers.
/// No allocation, no ownership.
struct TensorView {
    float* data = nullptr;
    int dims[4] = {0, 0, 0, 0};
    int ndim = 0;

    TensorView() = default;

    TensorView(float* d, int d0)
        : data(d), dims{d0, 0, 0, 0}, ndim(1) {}

    TensorView(float* d, int d0, int d1)
        : data(d), dims{d0, d1, 0, 0}, ndim(2) {}

    TensorView(float* d, int d0, int d1, int d2)
        : data(d), dims{d0, d1, d2, 0}, ndim(3) {}

    float& at(int i) {
        assert(ndim >= 1);
        return data[i];
    }

    const float& at(int i) const {
        assert(ndim >= 1);
        return data[i];
    }

    float& at(int i, int j) {
        assert(ndim >= 2);
        return data[i * dims[1] + j];
    }

    const float& at(int i, int j) const {
        assert(ndim >= 2);
        return data[i * dims[1] + j];
    }

    size_t numel() const {
        size_t n = 1;
        for (int i = 0; i < ndim; i++) n *= static_cast<size_t>(dims[i]);
        return n;
    }
};

}  // namespace nos
