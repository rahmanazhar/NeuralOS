/// @file platform_io_stub.cpp
/// @brief Placeholder for PlatformIO factory. Full implementation in Plan 03.

#include "platform_io.h"

namespace nos {

std::unique_ptr<PlatformIO> PlatformIO::create() {
    // Stub: will be implemented with pread fallback and io_uring in Plan 03.
    return nullptr;
}

}  // namespace nos
