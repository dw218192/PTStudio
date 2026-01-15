#pragma once
#include <core/legacy/utils.h>

namespace PTS {
struct Shader;
struct ShaderProgram;
DECL_ENUM(ShaderType, Vertex, Fragment);
constexpr auto to_string(ShaderType type) {
    switch (type) {
        case ShaderType::Vertex:
            return "Vertex Shader";
        case ShaderType::Fragment:
            return "Fragment Shader";
    }
    return "";
}
}  // namespace PTS