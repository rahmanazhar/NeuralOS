# ── NeuralOS TUI Dashboard ────────────────────────────────────────────────────

find_package(Curses REQUIRED)

add_library(nos_dashboard STATIC
    dashboard.cpp
    sparkline.cpp
)

target_link_libraries(nos_dashboard
    PUBLIC
        nos_server
        ${CURSES_LIBRARIES}
)

target_include_directories(nos_dashboard
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CURSES_INCLUDE_DIR}
)

nos_set_warnings(nos_dashboard)
