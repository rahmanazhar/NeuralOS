# ── Dependency Management ────────────────────────────────────────────────────
#
# Fetches external dependencies via FetchContent:
#   - Catch2 v3.9.1 (test framework)
#   - SentencePiece v0.2.1 (tokenizer)
#   - liburing >= 2.5 (Linux only, system library)

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# ── Catch2 ──────────────────────────────────────────────────────────────────

FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.9.1
)
FetchContent_MakeAvailable(catch2)

# ── SentencePiece ───────────────────────────────────────────────────────────
# Use FetchContent_Populate + add_subdirectory(EXCLUDE_FROM_ALL) to avoid
# installing SentencePiece targets alongside NeuralOS.

FetchContent_Declare(sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.1
)

FetchContent_GetProperties(sentencepiece)
if(NOT sentencepiece_POPULATED)
    FetchContent_Populate(sentencepiece)

    # SentencePiece build options: static-only, minimal build
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)

    add_subdirectory(
        ${sentencepiece_SOURCE_DIR}
        ${sentencepiece_BINARY_DIR}
        EXCLUDE_FROM_ALL
    )
endif()

# ── liburing (Linux only) ──────────────────────────────────────────────────

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(URING REQUIRED liburing>=2.5)
endif()
