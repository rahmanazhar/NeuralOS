/// @file sparkline.cpp
/// @brief Unicode sparkline renderer implementation.

#include "dashboard/sparkline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace nos {

// Unicode block elements: U+2581 through U+2588 (8 levels)
// Each character is 3 bytes in UTF-8: E2 96 81..88
static const char* const SPARK_CHARS[] = {
    "\xe2\x96\x81",  // U+2581 LOWER ONE EIGHTH BLOCK
    "\xe2\x96\x82",  // U+2582 LOWER ONE QUARTER BLOCK
    "\xe2\x96\x83",  // U+2583 LOWER THREE EIGHTHS BLOCK
    "\xe2\x96\x84",  // U+2584 LOWER HALF BLOCK
    "\xe2\x96\x85",  // U+2585 LOWER FIVE EIGHTHS BLOCK
    "\xe2\x96\x86",  // U+2586 LOWER THREE QUARTERS BLOCK
    "\xe2\x96\x87",  // U+2587 LOWER SEVEN EIGHTHS BLOCK
    "\xe2\x96\x88",  // U+2588 FULL BLOCK
};

std::string render_sparkline(const float* values, size_t count, size_t width) {
    if (values == nullptr || count == 0 || width == 0) return "";

    // Determine the range of values to render (last `width` values from the array)
    size_t start = 0;
    size_t render_count = count;
    if (count > width) {
        start = count - width;
        render_count = width;
    }

    // Clamp negative values to 0 and find min/max
    float vmin = 0.0f;
    float vmax = 0.0f;
    bool first = true;
    for (size_t i = 0; i < render_count; ++i) {
        float v = std::max(0.0f, values[start + i]);
        if (first) {
            vmin = v;
            vmax = v;
            first = false;
        } else {
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
    }

    std::string result;
    result.reserve(render_count * 3);  // Each UTF-8 char is 3 bytes

    float range = vmax - vmin;
    for (size_t i = 0; i < render_count; ++i) {
        float v = std::max(0.0f, values[start + i]);
        int level;
        if (range < 1e-6f) {
            // All values equal (or all zero) -- use middle bar
            level = 3;
        } else {
            float normalized = (v - vmin) / range;
            level = static_cast<int>(normalized * 7.0f);
            level = std::clamp(level, 0, 7);
        }
        result += SPARK_CHARS[level];
    }

    // If we had fewer values than width, pad with spaces
    for (size_t i = render_count; i < width; ++i) {
        result += ' ';
    }

    return result;
}

}  // namespace nos
