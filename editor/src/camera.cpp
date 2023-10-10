#include "include/camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

Camera::Camera(float fovy, unsigned px_width, unsigned px_height, Transform const& view) noexcept
    : m_view(view), m_fov(fovy),
		m_px_width(px_width), m_px_height(px_height),
		m_aspect(px_width / static_cast<float>(px_height))
{
    m_projection = glm::perspective(glm::radians(fovy), m_aspect, k_near, k_far);
    update_matrices();
}

auto Camera::ndc_to_wrold(glm::vec2 ndc, float z) const noexcept -> glm::vec3 {
    auto ret = m_inv_view_proj * glm::vec4{ ndc, z, 1.0f };
    ret /= ret.w;
    return ret;
}

auto Camera::world_to_ndc(glm::vec3 world) const noexcept -> glm::vec3 {
    auto ret = m_view_proj * glm::vec4(world, 1.0f);
    ret /= ret.w;
    return ret;
}

auto Camera::world_to_viewport(glm::vec3 world) const noexcept -> glm::vec2 {
    auto ndc = world_to_ndc(world);
    // convert to (0, 1) and invert y
    ndc = (ndc + 1.0f) / 2.0f;
    ndc.y = 1 - ndc.y;
    return glm::vec2{
        ndc.x * m_px_width, ndc.y * m_px_height
    };
}

auto Camera::viewport_to_world(glm::vec2 screen, float z) const noexcept -> glm::vec3 {
    auto ndc = glm::vec2{
        screen.x / m_px_width, screen.y / m_px_height
    };
    // convert to (-1, 1) and invert y
    ndc = ndc * 2.0f - 1.0f;
    ndc.y = 1 - ndc.y;
    return ndc_to_wrold(ndc, z);
}

void Camera::set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept {
    m_view.set_rotation(space, rot);
    update_matrices();
}

void Camera::set_position(TransformSpace space, glm::vec3 const& pos) noexcept {
    m_view.set_position(space, pos);
    update_matrices();
}

void Camera::set_view_transform(Transform const& transform) noexcept {
    m_view = transform;
    update_matrices();
}

void Camera::set_fov(float fov) noexcept {
    m_fov = fov;
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    update_matrices();
}

void Camera::update_matrices() noexcept {
    m_view_proj = m_projection * m_view.get_matrix();
    m_inv_view_proj = glm::inverse(m_view_proj);
}
