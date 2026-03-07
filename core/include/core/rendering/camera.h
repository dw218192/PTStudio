#pragma once

#include <glm/glm.hpp>

namespace pts::rendering {

class OrbitCamera {
   public:
    OrbitCamera();

    // ── Configuration ──
    void set_target(glm::vec3 target);
    void set_distance(float distance);
    void set_fov_y(float fov_degrees);
    void set_clip_planes(float near, float far);

    // ── Interaction ──
    /// Orbit: rotate around target. dx/dy are normalized deltas (e.g. mouse delta / viewport size).
    void orbit(float dx, float dy);

    /// Pan: translate target in the camera's local XY plane.
    void pan(float dx, float dy);

    /// Zoom: change distance to target. delta > 0 zooms in.
    void zoom(float delta);

    // ── Output ──
    [[nodiscard]] auto view_matrix() const -> glm::mat4;
    [[nodiscard]] auto projection_matrix(float aspect_ratio) const -> glm::mat4;

    // ── Accessors ──
    [[nodiscard]] auto target() const -> glm::vec3;
    [[nodiscard]] auto position() const -> glm::vec3;
    [[nodiscard]] auto distance() const -> float;
    [[nodiscard]] auto fov_y_degrees() const -> float;
    [[nodiscard]] auto near_plane() const -> float;
    [[nodiscard]] auto far_plane() const -> float;

   private:
    glm::vec3 m_target{0.0f, 0.0f, 0.0f};
    float m_distance = 5.0f;
    float m_yaw = 0.0f;     // radians, around Y axis
    float m_pitch = 0.3f;   // radians, elevation (clamped to avoid gimbal lock)
    float m_fov_y = 60.0f;  // degrees
    float m_near = 0.1f;
    float m_far = 1000.0f;

    // Tuning constants
    static constexpr float k_orbit_speed = 3.0f;
    static constexpr float k_pan_speed = 2.0f;
    static constexpr float k_zoom_speed = 0.15f;
    static constexpr float k_min_distance = 0.1f;
    static constexpr float k_max_distance = 500.0f;
    static constexpr float k_max_pitch = 1.5f;  // ~86 degrees, avoid flipping
};

}  // namespace pts::rendering
