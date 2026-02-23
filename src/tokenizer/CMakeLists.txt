# -- nos_tokenizer: SentencePiece tokenizer wrapper --

add_library(nos_tokenizer STATIC
    tokenizer.cpp
)

target_include_directories(nos_tokenizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

nos_set_warnings(nos_tokenizer)

# Link with SentencePiece
target_link_libraries(nos_tokenizer
    PUBLIC sentencepiece-static
)

# SentencePiece does not export its include directories via PUBLIC.
# We need to add the SentencePiece source directory manually.
target_include_directories(nos_tokenizer
    PRIVATE ${sentencepiece_SOURCE_DIR}/src
)
