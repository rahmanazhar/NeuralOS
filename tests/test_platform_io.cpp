/// @file test_platform_io.cpp
/// @brief Tests for PlatformIO backends (PreadBackend, factory, io_uring).

#include <catch2/catch_test_macros.hpp>

#include "platform_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: RAII temporary file with known content
// ---------------------------------------------------------------------------

class TempFile {
public:
    explicit TempFile(const std::string& content) {
        // Create a temporary file
        std::snprintf(path_, sizeof(path_), "/tmp/nos_io_test_XXXXXX");
        fd_ = ::mkstemp(path_);
        REQUIRE(fd_ >= 0);

        // Write content
        ssize_t written = ::write(fd_, content.data(), content.size());
        REQUIRE(written == static_cast<ssize_t>(content.size()));

        // Seek back (not strictly needed since we use pread, but for safety)
        ::lseek(fd_, 0, SEEK_SET);
    }

    explicit TempFile(const std::vector<uint8_t>& data) {
        std::snprintf(path_, sizeof(path_), "/tmp/nos_io_test_XXXXXX");
        fd_ = ::mkstemp(path_);
        REQUIRE(fd_ >= 0);

        ssize_t written = ::write(fd_, data.data(), data.size());
        REQUIRE(written == static_cast<ssize_t>(data.size()));

        ::lseek(fd_, 0, SEEK_SET);
    }

    ~TempFile() {
        if (fd_ >= 0) ::close(fd_);
        ::unlink(path_);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    int fd() const { return fd_; }
    const char* path() const { return path_; }

private:
    int fd_ = -1;
    char path_[256] = {};
};

// ===========================================================================
// PreadBackend Tests
// ===========================================================================

TEST_CASE("PreadBackend: basic read", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("Hello, NeuralOS!");

    char buf[32] = {};
    int rc = backend.submit_read(tmp.fd(), buf, 16, 0, nullptr);
    REQUIRE(rc == 0);
    REQUIRE(backend.pending() == 1);

    nos::IoCompletion comp{};
    int n = backend.poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result == 16);
    REQUIRE(std::string(buf, 16) == "Hello, NeuralOS!");
    REQUIRE(backend.pending() == 0);
}

TEST_CASE("PreadBackend: partial read", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("Hello, NeuralOS!");

    char buf[8] = {};
    int rc = backend.submit_read(tmp.fd(), buf, 5, 0, nullptr);
    REQUIRE(rc == 0);

    nos::IoCompletion comp{};
    int n = backend.poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result == 5);
    REQUIRE(std::string(buf, 5) == "Hello");
}

TEST_CASE("PreadBackend: offset read", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("Hello, NeuralOS!");

    char buf[16] = {};
    int rc = backend.submit_read(tmp.fd(), buf, 7, 7, nullptr);
    REQUIRE(rc == 0);

    nos::IoCompletion comp{};
    int n = backend.poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result == 7);
    REQUIRE(std::string(buf, 7) == "NeuralO");
}

TEST_CASE("PreadBackend: multiple reads before poll", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("Hello, NeuralOS!");

    // Submit 3 reads at different offsets
    char buf0[8] = {};
    char buf1[8] = {};
    char buf2[8] = {};

    REQUIRE(backend.submit_read(tmp.fd(), buf0, 5, 0, nullptr) == 0);
    REQUIRE(backend.submit_read(tmp.fd(), buf1, 7, 7, nullptr) == 0);
    REQUIRE(backend.submit_read(tmp.fd(), buf2, 1, 15, nullptr) == 0);
    REQUIRE(backend.pending() == 3);

    nos::IoCompletion comps[4] = {};
    int n = backend.poll(comps, 4, 0);
    REQUIRE(n == 3);
    REQUIRE(comps[0].result == 5);
    REQUIRE(std::string(buf0, 5) == "Hello");
    REQUIRE(comps[1].result == 7);
    REQUIRE(std::string(buf1, 7) == "NeuralO");
    REQUIRE(comps[2].result == 1);
    REQUIRE(std::string(buf2, 1) == "!");
    REQUIRE(backend.pending() == 0);
}

TEST_CASE("PreadBackend: invalid fd", "[io][pread]") {
    nos::PreadBackend backend;

    char buf[16] = {};
    int rc = backend.submit_read(-1, buf, 16, 0, nullptr);
    REQUIRE(rc == 0);  // submit itself succeeds; error is in the completion

    nos::IoCompletion comp{};
    int n = backend.poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result < 0);  // negative errno (e.g., -EBADF)
}

TEST_CASE("PreadBackend: read past EOF", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("short");

    char buf[16] = {};
    // Offset is beyond the file size (5 bytes)
    int rc = backend.submit_read(tmp.fd(), buf, 16, 1000, nullptr);
    REQUIRE(rc == 0);

    nos::IoCompletion comp{};
    int n = backend.poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result == 0);  // EOF: pread returns 0 bytes
}

TEST_CASE("PreadBackend: zero-length read", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("some data");

    char buf[1] = {};
    int rc = backend.submit_read(tmp.fd(), buf, 0, 0, nullptr);
    REQUIRE(rc == 0);

    nos::IoCompletion comp{};
    int n = backend.poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result == 0);  // zero-length read returns 0
}

TEST_CASE("PreadBackend: large read (1MB with pattern)", "[io][pread]") {
    nos::PreadBackend backend;

    // Create a 1MB file with a repeating pattern
    const size_t file_size = 1024 * 1024;  // 1 MB
    std::vector<uint8_t> pattern(file_size);
    for (size_t i = 0; i < file_size; ++i) {
        pattern[i] = static_cast<uint8_t>(i % 251);  // prime modulus for variety
    }
    TempFile tmp(pattern);

    // Read in 64KB chunks
    const size_t chunk_size = 64 * 1024;
    const int num_chunks = static_cast<int>(file_size / chunk_size);
    std::vector<std::vector<uint8_t>> buffers(num_chunks,
                                              std::vector<uint8_t>(chunk_size));

    for (int i = 0; i < num_chunks; ++i) {
        off_t offset = static_cast<off_t>(i * chunk_size);
        int rc = backend.submit_read(tmp.fd(), buffers[i].data(),
                                     chunk_size, offset, nullptr);
        REQUIRE(rc == 0);
    }
    REQUIRE(backend.pending() == num_chunks);

    // Poll all completions
    std::vector<nos::IoCompletion> comps(num_chunks);
    int n = backend.poll(comps.data(), num_chunks, 0);
    REQUIRE(n == num_chunks);

    // Verify each chunk matches the expected pattern
    for (int i = 0; i < num_chunks; ++i) {
        REQUIRE(comps[i].result == static_cast<int>(chunk_size));
        size_t base_offset = static_cast<size_t>(i) * chunk_size;
        for (size_t j = 0; j < chunk_size; ++j) {
            REQUIRE(buffers[i][j] ==
                    static_cast<uint8_t>((base_offset + j) % 251));
        }
    }
}

TEST_CASE("PreadBackend: userdata round-trip", "[io][pread]") {
    nos::PreadBackend backend;
    TempFile tmp("userdata test");

    int tag_a = 42;
    int tag_b = 99;
    int tag_c = 7;
    char buf0[4] = {};
    char buf1[4] = {};
    char buf2[4] = {};

    REQUIRE(backend.submit_read(tmp.fd(), buf0, 4, 0, &tag_a) == 0);
    REQUIRE(backend.submit_read(tmp.fd(), buf1, 4, 4, &tag_b) == 0);
    REQUIRE(backend.submit_read(tmp.fd(), buf2, 4, 8, &tag_c) == 0);

    nos::IoCompletion comps[3] = {};
    int n = backend.poll(comps, 3, 0);
    REQUIRE(n == 3);

    // Verify each completion carries the correct userdata pointer back
    REQUIRE(comps[0].userdata == &tag_a);
    REQUIRE(comps[1].userdata == &tag_b);
    REQUIRE(comps[2].userdata == &tag_c);

    // And the values behind those pointers are still correct
    REQUIRE(*static_cast<int*>(comps[0].userdata) == 42);
    REQUIRE(*static_cast<int*>(comps[1].userdata) == 99);
    REQUIRE(*static_cast<int*>(comps[2].userdata) == 7);
}

// ===========================================================================
// Factory Tests
// ===========================================================================

TEST_CASE("Factory: create() returns non-null", "[io][factory]") {
    auto io = nos::PlatformIO::create();
    REQUIRE(io != nullptr);
}

TEST_CASE("Factory: basic read through factory", "[io][factory]") {
    auto io = nos::PlatformIO::create();
    REQUIRE(io != nullptr);

    TempFile tmp("Factory works!");

    char buf[16] = {};
    int rc = io->submit_read(tmp.fd(), buf, 14, 0, nullptr);
    REQUIRE(rc == 0);

    nos::IoCompletion comp{};
    int n = io->poll(&comp, 1, 0);
    REQUIRE(n == 1);
    REQUIRE(comp.result == 14);
    REQUIRE(std::string(buf, 14) == "Factory works!");
}
