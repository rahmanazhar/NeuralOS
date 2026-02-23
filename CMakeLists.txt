cmake_minimum_required(VERSION 3.23)
project(neuralos VERSION 0.1.0 LANGUAGES C CXX)

# ── C++ Standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build Settings ──────────────────────────────────────────────────────────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── CMake Modules ───────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(cmake/SimdDetection.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/FetchDependencies.cmake)

# ── Testing ─────────────────────────────────────────────────────────────────
enable_testing()

# ── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(tests)
