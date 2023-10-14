#pragma once
#include "ext.h"
#include "transform.h"
#include "ray.h"

struct Camera {
    Camera(float fovy, unsigned px_width, unsigned px_height, Transform const& view) noexcept;
    
    [[nodiscard]] auto get_view_proj() const noexcept -> glm::mat4 const& { return m_view_proj; }
    [[nodiscard]] auto get_view() const noexcept -> glm::mat4 const& { return m_view.get_matrix(); }
    [[nodiscard]] auto get_view_transform() const noexcept -> Transform const& { return m_view; }
    [[nodiscard]] auto get_projection() const noexcept -> glm::mat4 const& { return m_projection; }
    [[nodiscard]] auto get_fov() const noexcept { return m_fov; }

    /**
     * \brief Converts a point in screen space to normalized device coordinates
     * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
     * \return the point in normalized device coordinates, OpenGL standard, i.e. [-1, 1] for x,y,z
     */
    [[nodiscard]] auto screen_to_ndc(glm::vec2 screen) const noexcept -> glm::vec2;

    /**
     * \brief Converts a point in normalized device coordinates to world space
     * \param ndc the point in normalized device coordinates, OpenGL standard, i.e. [-1, 1] for x,y,z
     * \param z the z value to use for the point
     * \return the point in world space
     */
    [[nodiscard]] auto ndc_to_wrold(glm::vec2 ndc, float z = k_near) const noexcept -> glm::vec3;

    /**
     * \brief Converts a point in world space to normalized device coordinates
     * \param world the point in world space
     * \return the point in normalized device coordinates, OpenGL standard, i.e. [-1, 1] for x,y,z
    */
    [[nodiscard]] auto world_to_ndc(glm::vec3 world) const noexcept -> glm::vec3;

    /**
     * \brief Converts a point in world space to screen space
     * \param world the point in world space
     * \return the point in screen space; top left is (0,0), bottom right is (width, height)
     */
    [[nodiscard]] auto world_to_viewport(glm::vec3 world) const noexcept -> glm::vec2;
    /**
     * \brief Converts a point in screen space to world space
     * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
     * \param z the z value to use for the point
     * \return the point in world space
     */
    [[nodiscard]] auto viewport_to_world(glm::vec2 screen, float z = k_near) const noexcept -> glm::vec3;

    /**
     * \brief Converts a point in screen space to a ray in world space
     * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
     * \return the ray in world space
    */
    [[nodiscard]] auto viewport_to_ray(glm::vec2 screen) const noexcept -> Ray;

    void set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept;
    void set_position(TransformSpace space, glm::vec3 const& pos) noexcept;
    void set_view_transform(Transform const& transform) noexcept;

	void set_fov(float fov) noexcept;
    void set_viewport(unsigned px_width, unsigned px_height) noexcept;
private:
    void update_matrices() noexcept;

    static constexpr float k_near = 0.1f, k_far = 100000.0f;
    Transform m_view;
    float m_fov, m_aspect, m_px_width, m_px_height;
    glm::mat4 m_projection;
    glm::mat4 m_view_proj;
    glm::mat4 m_inv_view_proj;
}; 