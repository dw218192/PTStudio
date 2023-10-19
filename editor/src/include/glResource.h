#pragma once
#include <GL/glew.h>
#include <memory>

struct GLResourceDeleter;

template<typename T>
using GLResRef = std::unique_ptr<T, GLResourceDeleter>;

struct GLResource {
    friend struct GLResourceDeleter;

    [[nodiscard]] auto valid() const noexcept -> bool { return m_handle != 0; }
    [[nodiscard]] auto handle() const noexcept -> GLuint { return m_handle; }

    GLResource(GLResource&& other) noexcept;
    GLResource(GLResource const&) = delete;

    GLResource& operator=(GLResource&& other) noexcept;
    GLResource& operator=(GLResource const&) = delete;

protected:
    void swap(GLResource&& other) noexcept;
    GLResource(GLuint handle) noexcept { m_handle = handle; }
    GLResource() noexcept = default;
    virtual ~GLResource() noexcept = default;
    GLuint m_handle { 0 };
};

struct GLResourceDeleter {
	void operator()(GLResource const* res) const {
        delete res;
	}
};