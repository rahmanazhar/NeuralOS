#pragma once

/// @file sparkline.h
/// @brief Unicode sparkline renderer using block characters U+2581..U+2588.

#include <cstddef>
#include <string>

namespace nos {

/// Render a sparkline from float values using Unicode block characters.
/// Maps min..max of the input range to 8 sparkline levels (U+2581..U+2588).
/// @param values  Pointer to float array (may contain negative values, treated as 0)
/// @param count   Number of values in array
/// @param width   Desired output width in characters (uses last `width` values if count > width)
/// @return UTF-8 string of `width` sparkline characters. Empty string if count == 0 or values == nullptr.
std::string render_sparkline(const float* values, size_t count, size_t width);

}  // namespace nos
