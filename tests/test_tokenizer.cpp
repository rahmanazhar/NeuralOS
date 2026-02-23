/// @file test_tokenizer.cpp
/// @brief Tokenizer encode/decode round-trip tests.
///
/// Tier 1 (always runs): error handling tests that don't require a .model file.
/// Tier 2 (conditional): functional tests that require a real SentencePiece model.
/// Set NOS_TEST_TOKENIZER_MODEL=/path/to/llama2.model to enable Tier 2.

#include <catch2/catch_test_macros.hpp>

#include "tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Tier 1: Always runs (no model file needed) ─────────────────────────────

TEST_CASE("Tokenizer load failure -- non-existent path", "[tokenizer][tier1]") {
    nos::Tokenizer tok;
    REQUIRE_FALSE(tok.load("/nonexistent/path/to/model.model"));
    REQUIRE_FALSE(tok.is_loaded());
}

TEST_CASE("Tokenizer load failure -- invalid file", "[tokenizer][tier1]") {
    // Create a temp file with garbage bytes
    auto tmp_path = fs::temp_directory_path() / "garbage_model.model";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f << "this is not a valid sentencepiece model file at all!";
        f.close();
    }

    nos::Tokenizer tok;
    REQUIRE_FALSE(tok.load(tmp_path.string()));
    REQUIRE_FALSE(tok.is_loaded());

    fs::remove(tmp_path);
}

TEST_CASE("Tokenizer unloaded usage -- graceful behavior", "[tokenizer][tier1]") {
    nos::Tokenizer tok;
    REQUIRE_FALSE(tok.is_loaded());

    SECTION("encode returns empty") {
        auto ids = tok.encode("Hello, world!");
        REQUIRE(ids.empty());
    }

    SECTION("decode returns empty") {
        auto text = tok.decode({1, 2, 3});
        REQUIRE(text.empty());
    }

    SECTION("vocab_size returns 0") {
        REQUIRE(tok.vocab_size() == 0);
    }

    SECTION("bos_id returns -1") {
        REQUIRE(tok.bos_id() == -1);
    }

    SECTION("eos_id returns -1") {
        REQUIRE(tok.eos_id() == -1);
    }

    SECTION("pad_id returns -1") {
        REQUIRE(tok.pad_id() == -1);
    }

    SECTION("unk_id returns -1") {
        REQUIRE(tok.unk_id() == -1);
    }
}

// ── Tier 2: Requires NOS_TEST_TOKENIZER_MODEL ──────────────────────────────

static std::string get_model_path() {
    const char* env = std::getenv("NOS_TEST_TOKENIZER_MODEL");
    if (env == nullptr || std::string(env).empty()) {
        return {};
    }
    return std::string(env);
}

TEST_CASE("Tokenizer load success", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set -- skipping Tier 2 tokenizer tests. "
             "Set NOS_TEST_TOKENIZER_MODEL=/path/to/llama2.model to enable.");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));
    REQUIRE(tok.is_loaded());
    REQUIRE(tok.vocab_size() > 0);

    std::cout << "[test_tokenizer] Model: " << model_path << std::endl;
    std::cout << "[test_tokenizer] Vocab size: " << tok.vocab_size() << std::endl;
}

TEST_CASE("Tokenizer encode-decode round-trip (ASCII)", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));

    std::string input = "Hello, world!";
    auto ids = tok.encode(input);
    REQUIRE_FALSE(ids.empty());

    auto output = tok.decode(ids);
    REQUIRE(output == input);
}

TEST_CASE("Tokenizer encode-decode round-trip (Unicode)", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));

    // Accented characters and basic CJK
    std::string input = "cafe\xCC\x81 \xE4\xBD\xA0\xE5\xA5\xBD";  // "cafe\u0301 \u4F60\u597D"
    auto ids = tok.encode(input);
    REQUIRE_FALSE(ids.empty());

    auto output = tok.decode(ids);
    REQUIRE(output == input);
}

TEST_CASE("Tokenizer empty string", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));

    SECTION("Encode empty string") {
        auto ids = tok.encode("");
        REQUIRE(ids.empty());
    }

    SECTION("Decode empty token list") {
        auto text = tok.decode({});
        REQUIRE(text.empty());
    }
}

TEST_CASE("Tokenizer special tokens", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));

    SECTION("bos_id is valid") {
        int bos = tok.bos_id();
        REQUIRE(bos >= 0);
        REQUIRE(bos < tok.vocab_size());
    }

    SECTION("eos_id is valid") {
        int eos = tok.eos_id();
        REQUIRE(eos >= 0);
        REQUIRE(eos < tok.vocab_size());
    }
}

TEST_CASE("Tokenizer deterministic encoding", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));

    std::string text = "The quick brown fox jumps over the lazy dog.";
    auto ids1 = tok.encode(text);
    auto ids2 = tok.encode(text);
    REQUIRE(ids1 == ids2);
}

TEST_CASE("Tokenizer long text round-trip", "[tokenizer][tier2]") {
    auto model_path = get_model_path();
    if (model_path.empty()) {
        SKIP("NOS_TEST_TOKENIZER_MODEL not set");
    }

    nos::Tokenizer tok;
    REQUIRE(tok.load(model_path));

    // 200+ word paragraph
    std::string paragraph =
        "In the vast expanse of the digital frontier, artificial intelligence has emerged "
        "as one of the most transformative technologies of our time. From natural language "
        "processing to computer vision, from robotics to drug discovery, AI systems are "
        "reshaping every facet of human endeavor. The development of large language models "
        "has particularly captured the public imagination, demonstrating remarkable abilities "
        "in understanding and generating human text. These models, trained on enormous "
        "corpora of text data, can engage in conversation, write creative fiction, analyze "
        "complex documents, and assist with programming tasks. However, the computational "
        "demands of these models present significant challenges. Running inference on models "
        "with hundreds of billions of parameters requires substantial hardware resources, "
        "often involving multiple high-end GPUs or specialized AI accelerators. This has "
        "led researchers to explore various optimization techniques, including quantization, "
        "pruning, knowledge distillation, and mixture-of-experts architectures. The goal "
        "is to make these powerful models more accessible, allowing them to run on consumer "
        "hardware with limited memory and processing power. By leveraging techniques like "
        "ternary quantization and efficient expert paging, it becomes possible to achieve "
        "near-frontier performance on devices that would otherwise be incapable of running "
        "such models. This democratization of AI technology has profound implications for "
        "education, healthcare, scientific research, and creative expression worldwide.";

    auto ids = tok.encode(paragraph);
    REQUIRE(ids.size() > 100);  // Should produce many tokens for this text

    auto output = tok.decode(ids);
    REQUIRE(output == paragraph);
}
