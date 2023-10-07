#include "include/camera.h"

#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(float fovy, float aspect, Transform const& transform) noexcept 
    : m_transform(transform), m_fov(fovy), m_aspect(aspect)
{
    m_projection = glm::perspective(glm::radians(fovy), aspect, k_near, k_far);
    m_view_proj = m_projection * m_transform.get_matrix();
}

void Camera::set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept {
    m_transform.set_rotation(space, rot);
    m_view_proj = m_projection * m_transform.get_matrix();
}

void Camera::set_position(TransformSpace space, glm::vec3 const& pos) noexcept {
    m_transform.set_position(space, pos);
    m_view_proj = m_projection * m_transform.get_matrix();
}

void Camera::set_transform(Transform const& transform) noexcept {
    m_transform = transform;
    m_view_proj = m_projection * m_transform.get_matrix();
}

void Camera::set_fov(float fov) noexcept {
    m_fov = fov;
    m_projection = glm::perspective(glm::radians(m_fov), m_aspect, k_near, k_far);
    m_view_proj = m_projection * m_transform.get_matrix();
}
