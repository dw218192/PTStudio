#pragma once
#include "ext.h"
#include "transform.h"

struct Camera {
    Camera(float fovy, float aspect, Transform const& transform) noexcept;
    
    [[nodiscard]] auto get_view_proj() const noexcept -> glm::mat4 const& { return m_view_proj; }
    [[nodiscard]] auto get_transform() const noexcept -> Transform const& { return m_transform; }
    [[nodiscard]] auto get_projection() const noexcept -> glm::mat4 const& { return m_projection; }
    [[nodiscard]] auto get_fov() const noexcept { return m_fov; }

    void set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept;
    void set_position(TransformSpace space, glm::vec3 const& pos) noexcept;
    void set_transform(Transform const& transform) noexcept;
    void set_fov(float fov) noexcept;

private:
    static constexpr float k_near = 0.1f, k_far = 100000.0f;
    Transform m_transform;
    float m_fov, m_aspect;
    glm::mat4 m_projection;
    glm::mat4 m_view_proj;
}; 