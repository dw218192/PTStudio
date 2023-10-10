#pragma once
#include "ext.h"
#include "transform.h"

struct Camera {
    Camera(float fovy, unsigned px_width, unsigned px_height, Transform const& view) noexcept;
    
    [[nodiscard]] auto get_view_proj() const noexcept -> glm::mat4 const& { return m_view_proj; }
    [[nodiscard]] auto get_view() const noexcept -> glm::mat4 const& { return m_view.get_matrix(); }
    [[nodiscard]] auto get_projection() const noexcept -> glm::mat4 const& { return m_projection; }
    [[nodiscard]] auto get_fov() const noexcept { return m_fov; }

    [[nodiscard]] auto ndc_to_wrold(glm::vec2 ndc, float z = 1.0f) const noexcept -> glm::vec3;
    [[nodiscard]] auto world_to_ndc(glm::vec3 world) const noexcept -> glm::vec3;
    [[nodiscard]] auto world_to_viewport(glm::vec3 world) const noexcept -> glm::vec2;
    [[nodiscard]] auto viewport_to_world(glm::vec2 screen, float z = 1.0f) const noexcept -> glm::vec3;


    void set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept;
    void set_position(TransformSpace space, glm::vec3 const& pos) noexcept;
    void set_view_transform(Transform const& transform) noexcept;

	void set_fov(float fov) noexcept;

private:
    void update_matrices() noexcept;

    static constexpr float k_near = 0.1f, k_far = 100000.0f;
    Transform m_view;
    float m_fov, m_aspect, m_px_width, m_px_height;
    glm::mat4 m_projection;
    glm::mat4 m_view_proj;
    glm::mat4 m_inv_view_proj;
}; 