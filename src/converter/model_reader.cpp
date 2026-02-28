/// @file model_reader.cpp
/// @brief ModelReader factory implementation.

#include "converter/model_reader.h"
#include "converter/safetensors_reader.h"
#include "converter/gguf_reader.h"

#include <cstring>
#include <filesystem>

#include <fcntl.h>
#include <unistd.h>

namespace nos {

std::unique_ptr<ModelReader> ModelReader::create(const std::string& path) {
    namespace fs = std::filesystem;

    // Check if it's a directory (SafeTensors model)
    if (fs::is_directory(path)) {
        auto reader = std::make_unique<SafeTensorsReader>();
        if (reader->open(path)) return reader;
        return nullptr;
    }

    // Detect format by extension or magic bytes
    if (fs::is_regular_file(path)) {
        std::string ext = fs::path(path).extension().string();

        if (ext == ".safetensors") {
            auto reader = std::make_unique<SafeTensorsReader>();
            if (reader->open(path)) return reader;
            return nullptr;
        }

        if (ext == ".gguf") {
            auto reader = std::make_unique<GgufReader>();
            if (reader->open(path)) return reader;
            return nullptr;
        }

        // Try detecting by magic bytes
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            char magic[4];
            if (::pread(fd, magic, 4, 0) == 4) {
                ::close(fd);
                if (std::memcmp(magic, "GGUF", 4) == 0) {
                    auto reader = std::make_unique<GgufReader>();
                    if (reader->open(path)) return reader;
                }
            } else {
                ::close(fd);
            }
        }

        // Default: try SafeTensors
        auto reader = std::make_unique<SafeTensorsReader>();
        if (reader->open(path)) return reader;
    }

    return nullptr;
}

}  // namespace nos
