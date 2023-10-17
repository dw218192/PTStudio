#pragma once
#include <GL/glew.h>
#include <tl/expected.hpp>
#include <string>

template<typename Derived>
struct GLResource {
    [[nodiscard]] auto valid() const noexcept -> bool { return static_cast<Derived const&>(*this).m_handle != 0; }
    [[nodiscard]] auto handle() const noexcept -> GLuint { return static_cast<Derived const&>(*this).m_handle; }

    GLResource() noexcept = default;
    GLResource(GLResource const&) = delete;
	GLResource(GLResource&& other) noexcept : m_handle{other.m_handle} {
        other.m_handle = 0;
    }
protected:
    GLuint m_handle{ 0 };
};