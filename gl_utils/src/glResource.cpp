#include "glResource.h"

using namespace PTS;

GLResource::GLResource(GLResource&& other) noexcept {
    swap(std::move(other));
}

auto GLResource::operator=(GLResource&& other) noexcept -> GLResource& {
    swap(std::move(other));
    return *this;
}

GLResource::GLResource(GLResource const& other) noexcept {
    m_handle = other.m_handle;
}

auto GLResource::operator=(GLResource const& other) noexcept -> GLResource& {
    m_handle = other.m_handle;
    return *this;
}

auto GLResource::swap(GLResource&& other) noexcept -> void {
    m_handle = other.m_handle;
    other.m_handle = 0;
}
