# ── NeuralOS HTTP Server ──────────────────────────────────────────────────────

add_library(nos_server STATIC
    http_server.cpp
    chat_template.cpp
)

target_link_libraries(nos_server
    PUBLIC
        neuralos_shared
        httplib::httplib
        nlohmann_json::nlohmann_json
)

target_include_directories(nos_server
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

nos_set_warnings(nos_server)
