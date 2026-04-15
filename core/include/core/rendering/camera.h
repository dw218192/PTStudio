#pragma once

#include <core/rendering/upAxis.h>

#include <glm/glm.hpp>

namespace pts::rendering {

class OrbitCamera {
   public:
    OrbitCamera();

    // -- Configuration --
    void set_target(glm::vec3 target);
    void set_distance(float distance);
    void set_fov_y(float fov_degrees);
    void set_yaw(float radians);
    void set_pitch(float radians);
    void set_clip_planes(float near, float far);
    void set_up_axis(UpAxis axis);

    /// Adjust camera limits for the stage's unit system.
    /// Scales near/far planes, distance limits, and movement speed.
    void apply_meters_per_unit(float meters_per_unit);

    // -- Interaction --
    /// Orbit: rotate around target. dx/dy are normalized deltas (e.g. mouse delta / viewport size).
    void orbit(float dx, float dy);

    /// Pan: translate target in the camera's local XY plane.
    void pan(float dx, float dy);

    /// Zoom: change distance to target. delta > 0 zooms in.
    void zoom(float delta);

    /// Move: translate target along the camera's local axes.
    /// forward > 0 moves toward the look direction, right > 0 moves rightward, up > 0 moves upward.
    void move(float forward, float right, float up, float dt);

    // -- Output --
    [[nodiscard]] auto view_matrix() const -> glm::mat4;
    [[nodiscard]] auto projection_matrix(float aspect_ratio) const -> glm::mat4;

    // -- Accessors --
    [[nodiscard]] auto target() const -> glm::vec3;
    [[nodiscard]] auto position() const -> glm::vec3;
    [[nodiscard]] auto distance() const -> float;
    [[nodiscard]] auto fov_y_degrees() const -> float;
    [[nodiscard]] auto near_plane() const -> float;
    [[nodiscard]] auto far_plane() const -> float;
    [[nodiscard]] auto up_axis() const -> UpAxis;

   private:
    glm::vec3 m_target{0.0f, 0.0f, 0.0f};
    float m_distance = 5.0f;
    float m_yaw = 0.0f;     // radians, around Y axis
    float m_pitch = 0.3f;   // radians, elevation (clamped to avoid gimbal lock)
    float m_fov_y = 60.0f;  // degrees
    float m_near = 0.1f;
    float m_far = 1000.0f;
    UpAxis m_up_axis = UpAxis::Y;

    // Tuning constants (base values assume meters)
    static constexpr float k_orbit_speed = 3.0f;
    static constexpr float k_pan_speed = 2.0f;
    static constexpr float k_zoom_speed = 0.15f;
    static constexpr float k_max_pitch = 1.5f;  // ~86 degrees, avoid flipping

    // Unit-scaled limits (adjusted by apply_meters_per_unit)
    float m_min_distance = 0.1f;
    float m_max_distance = 500.0f;
    float m_move_speed = 5.0f;
};

}  // namespace pts::rendering
