#include "include/glResource.h"

GLResource::GLResource(GLResource&& other) noexcept {
    swap(std::move(other));
}

GLResource& GLResource::operator=(GLResource&& other) noexcept {
    swap(std::move(other));
    return *this;
}

GLResource::GLResource(GLResource& other) noexcept {
}

void GLResource::swap(GLResource&& other) noexcept {
    m_handle = other.m_handle;
    other.m_handle = 0;
}
