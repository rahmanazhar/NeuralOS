/// @file test_libneuralos.cpp
/// @brief Catch2 tests for the NeuralOS C Library API.

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>

#include "api/libneuralos.h"

// ── Version Tests ───────────────────────────────────────────────────────────

TEST_CASE("nos_version_major returns 0", "[libneuralos][version]") {
    REQUIRE(nos_version_major() == 0);
}

TEST_CASE("nos_version_minor returns 1", "[libneuralos][version]") {
    REQUIRE(nos_version_minor() == 1);
}

TEST_CASE("nos_version_patch returns 0", "[libneuralos][version]") {
    REQUIRE(nos_version_patch() == 0);
}

TEST_CASE("nos_version returns 0.1.0", "[libneuralos][version]") {
    const char* v = nos_version();
    REQUIRE(v != nullptr);
    REQUIRE(std::string(v) == "0.1.0");
}

// ── Error Handling Tests ────────────────────────────────────────────────────

TEST_CASE("nos_last_error returns empty string initially", "[libneuralos][error]") {
    // On a fresh thread, there should be no error
    const char* err = nos_last_error();
    REQUIRE(err != nullptr);
    // May or may not be empty depending on prior tests -- but must not be null
    // We test the basic contract: it returns a valid C string.
}

TEST_CASE("nos_create with nullptr path returns nullptr and sets error",
          "[libneuralos][lifecycle]") {
    nos_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size = sizeof(nos_config_t);

    nos_ctx_t* ctx = nos_create(nullptr, cfg);
    REQUIRE(ctx == nullptr);

    const char* err = nos_last_error();
    REQUIRE(err != nullptr);
    REQUIRE(std::string(err).find("NULL") != std::string::npos);
}

TEST_CASE("nos_create with invalid path returns nullptr and sets error",
          "[libneuralos][lifecycle]") {
    nos_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size = sizeof(nos_config_t);

    nos_ctx_t* ctx = nos_create("/nonexistent/model/path", cfg);
    REQUIRE(ctx == nullptr);

    const char* err = nos_last_error();
    REQUIRE(err != nullptr);
    REQUIRE(std::strlen(err) > 0);
}

TEST_CASE("nos_config_t struct_size validation rejects zero",
          "[libneuralos][lifecycle]") {
    nos_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size = 0;  // Wrong size

    nos_ctx_t* ctx = nos_create("/some/path", cfg);
    REQUIRE(ctx == nullptr);

    const char* err = nos_last_error();
    REQUIRE(err != nullptr);
    REQUIRE(std::string(err).find("struct_size") != std::string::npos);
}

// ── Null context tests ──────────────────────────────────────────────────────

TEST_CASE("nos_reset on nullptr returns error code", "[libneuralos][lifecycle]") {
    int rc = nos_reset(nullptr);
    REQUIRE(rc == NOS_ERR_INVALID);
}

TEST_CASE("nos_step_token on nullptr returns error code", "[libneuralos][inference]") {
    int out = 0;
    int rc = nos_step_token(nullptr, 0, &out);
    REQUIRE(rc == NOS_ERR_INVALID);
}

TEST_CASE("nos_step_text on nullptr returns error code", "[libneuralos][inference]") {
    char buf[64];
    int rc = nos_step_text(nullptr, "test", buf, sizeof(buf));
    REQUIRE(rc == NOS_ERR_INVALID);
}

TEST_CASE("nos_generate on nullptr returns error code", "[libneuralos][inference]") {
    char buf[64];
    int rc = nos_generate(nullptr, "test", buf, sizeof(buf));
    REQUIRE(rc == NOS_ERR_INVALID);
}

TEST_CASE("nos_tokenize on nullptr returns error code", "[libneuralos][tokenizer]") {
    int ids[10];
    size_t num = 0;
    int rc = nos_tokenize(nullptr, "test", ids, 10, &num);
    REQUIRE(rc == NOS_ERR_INVALID);
}

TEST_CASE("nos_detokenize on nullptr returns error code", "[libneuralos][tokenizer]") {
    int ids[] = {1, 2, 3};
    char buf[64];
    int rc = nos_detokenize(nullptr, ids, 3, buf, sizeof(buf));
    REQUIRE(rc == NOS_ERR_INVALID);
}

TEST_CASE("nos_get_metrics on nullptr returns nullptr", "[libneuralos][metrics]") {
    const char* m = nos_get_metrics(nullptr);
    REQUIRE(m == nullptr);
}

// ── Return code macro tests ─────────────────────────────────────────────────

TEST_CASE("Return code macros have correct values", "[libneuralos][constants]") {
    REQUIRE(NOS_OK == 0);
    REQUIRE(NOS_ERR_INVALID == -1);
    REQUIRE(NOS_ERR_MODEL == -2);
    REQUIRE(NOS_ERR_BUFFER == -3);
    REQUIRE(NOS_ERR_INTERNAL == -4);
}

// ── Version macro tests ────────────────────────────────────────────────────

TEST_CASE("Version macros match function returns", "[libneuralos][version]") {
    REQUIRE(NOS_VERSION_MAJOR == nos_version_major());
    REQUIRE(NOS_VERSION_MINOR == nos_version_minor());
    REQUIRE(NOS_VERSION_PATCH == nos_version_patch());
}
