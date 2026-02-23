# ── NeuralOS Tests ───────────────────────────────────────────────────────────

include(CTest)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

# ── Smoke Test ──────────────────────────────────────────────────────────────
add_executable(test_smoke test_smoke.cpp)

target_link_libraries(test_smoke
    PRIVATE
        Catch2::Catch2WithMain
        nos_format
)

target_include_directories(test_smoke
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/format
)

catch_discover_tests(test_smoke)
