#pragma once
#include <GL/glew.h>
#include <core/utils.h>

#include <memory>

namespace PTS {
struct GLResourceDeleter;

struct GLParam {
    GLenum param_name;
    GLint param_val;
};

template <typename T>
using UniqueGLResRef = std::unique_ptr<T, GLResourceDeleter>;

template <typename T>
using GLResRef = std::shared_ptr<T>;

struct GLResource {
    friend struct GLResourceDeleter;

    NODISCARD auto valid() const noexcept -> bool {
        return m_handle != 0;
    }
    NODISCARD auto handle() const noexcept -> GLuint {
        return m_handle;
    }

    GLResource(GLResource&& other) noexcept;
    auto operator=(GLResource&& other) noexcept -> GLResource&;
    GLResource(GLResource const& other) noexcept;
    auto operator=(GLResource const& other) noexcept -> GLResource&;

   protected:
    auto swap(GLResource&& other) noexcept -> void;
    GLResource(GLuint handle) noexcept {
        m_handle = handle;
    }
    GLResource() noexcept = default;
    virtual ~GLResource() noexcept = default;
    GLuint m_handle{0};
};

struct GLResourceDeleter {
    auto operator()(GLResource const* res) const -> void {
        delete res;
    }
};
}  // namespace PTS
