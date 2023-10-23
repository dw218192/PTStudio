#pragma once
#include <GL/glew.h>
#include <memory>

struct GLResourceDeleter;

struct GLParam {
    GLenum param_name;
    GLint param_val;
};

template<typename T>
using UniqueGLResRef = std::unique_ptr<T, GLResourceDeleter>;

template<typename T>
using GLResRef = std::shared_ptr<T>;

struct GLResource {
    friend struct GLResourceDeleter;

    [[nodiscard]] auto valid() const noexcept -> bool { return m_handle != 0; }
    [[nodiscard]] auto handle() const noexcept -> GLuint { return m_handle; }

    GLResource(GLResource&& other) noexcept;
    GLResource& operator=(GLResource&& other) noexcept;
    GLResource(GLResource& other) noexcept;
    GLResource& operator=(GLResource const& other) noexcept;
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