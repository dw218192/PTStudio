#pragma once
#include "ray.h"
#include "utils.h"

struct LookAtParams {
    glm::vec3 eye, center, up;
};

struct Camera {
    Camera(float fovy, unsigned px_width, unsigned px_height, glm::vec3 eye, glm::vec3 center, glm::vec3 up) noexcept;
    Camera(float fovy, unsigned px_width, unsigned px_height, LookAtParams const& params) noexcept;

    NODISCARD auto get_view_proj() const noexcept -> glm::mat4 const& { return m_view_proj; }
    NODISCARD auto get_view() const noexcept -> glm::mat4 const& { return m_view; }
    NODISCARD auto get_projection() const noexcept -> glm::mat4 const& { return m_projection; }
    NODISCARD auto get_fov() const noexcept { return m_fov; }

    /**
     * \brief Converts a point in screen space to normalized device coordinates
     * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
     * \return the point in normalized device coordinates, OpenGL standard, i.e. [-1, 1] for x,y,z
     */
    NODISCARD auto screen_to_ndc(glm::vec2 screen) const noexcept -> glm::vec2;

    /**
     * \brief Converts a point in normalized device coordinates to world space
     * \param ndc the point in normalized device coordinates, OpenGL standard, i.e. [-1, 1] for x,y,z
     * \param z the z value to use for the point
     * \return the point in world space
     */
    NODISCARD auto ndc_to_wrold(glm::vec2 ndc, float z) const noexcept -> glm::vec3;

    /**
     * \brief Converts a point in world space to normalized device coordinates
     * \param world the point in world space
     * \return the point in normalized device coordinates, OpenGL standard, i.e. [-1, 1] for x,y,z
    */
    NODISCARD auto world_to_ndc(glm::vec3 world) const noexcept -> glm::vec3;

    /**
     * \brief Converts a point in world space to screen space
     * \param world the point in world space
     * \return the point in screen space; top left is (0,0), bottom right is (width, height)
     */
    NODISCARD auto world_to_viewport(glm::vec3 world) const noexcept -> glm::vec2;
    /**
     * \brief Converts a point in screen space to world space
     * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
     * \param z the z value to use for the point
     * \return the point in world space
     */
    NODISCARD auto viewport_to_world(glm::vec2 screen, float z = -1.0f) const noexcept -> glm::vec3;

    /**
     * \brief Converts a point in screen space to a ray in world space
     * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
     * \return the ray in world space
    */
    NODISCARD auto viewport_to_ray(glm::vec2 screen) const noexcept -> Ray;

	void set_fov(float fov) noexcept;
    void set_viewport(unsigned px_width, unsigned px_height) noexcept;

    void set_delta_rotation(glm::vec3 const& delta) noexcept;
	void set_delta_dolly(glm::vec3 const& delta) noexcept;
    void set_delta_zoom(float delta) noexcept;

private:
    void update_matrices() noexcept;

    static constexpr float k_near = 0.1f, k_far = 100.0f;
    static constexpr float k_min_fov = 20.0f, k_max_fov = 120.0f;

    float m_fov, m_aspect, m_px_width, m_px_height;

    glm::vec3 m_eye, m_center, m_up;
    glm::mat4 m_cam_transform;
    glm::mat4 m_view;
    glm::mat4 m_projection;
    glm::mat4 m_view_proj;
    glm::mat4 m_inv_view_proj;
}; 