# ── nos_tokenizer: SentencePiece tokenizer wrapper ───────────────────────────

add_library(nos_tokenizer STATIC
    tokenizer_stub.cpp
)

target_include_directories(nos_tokenizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_tokenizer)

# Link with SentencePiece
target_link_libraries(nos_tokenizer
    PUBLIC sentencepiece-static
)
