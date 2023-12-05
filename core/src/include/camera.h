#pragma once
#include "ray.h"
#include "utils.h"
#include "reflection.h"

namespace PTS {
    struct LookAtParams {
        glm::vec3 eye{ 0,5,5 };
        glm::vec3 center{ 0, 0, 0 };
        glm::vec3 up{ 0, 1, 0 };
    };

    struct Camera {
        Camera() noexcept;
        Camera(float fovy, float aspect, glm::vec3 eye, glm::vec3 center, glm::vec3 up) noexcept;
        Camera(float fovy, float aspect, LookAtParams const& params) noexcept;

        NODISCARD auto get_view_proj() const noexcept -> glm::mat4 const& { return m_view_proj; }
        NODISCARD auto get_view() const noexcept -> glm::mat4 const& { return m_view; }
        NODISCARD auto get_projection() const noexcept -> glm::mat4 const& { return m_projection; }
        NODISCARD auto get_inv_view_proj() const noexcept -> glm::mat4 const& { return m_inv_view_proj; }
        NODISCARD auto get_fov() const noexcept { return m_fov; }
        NODISCARD auto get_aspect() const noexcept { return m_aspect; }

        /**
         * \brief Returns the position of the camera
         * \return the position of the camera
        */
        NODISCARD auto get_eye() const noexcept { return m_eye; }

        /**
         * \brief Returns the point that the camera is looking at
         * \return the point that the camera is looking at
        */
        NODISCARD auto get_center() const noexcept { return m_center; }
        NODISCARD auto get_up() const noexcept { return m_up; }

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
         * \param vp_size the size of viewport
         * \return the point in screen space; top left is (0,0), bottom right is (width, height)
         */
        NODISCARD auto world_to_viewport(glm::vec3 world, glm::ivec2 vp_size) const noexcept -> glm::vec2;
        /**
         * \brief Converts a point in screen space to world space
         * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
         * \param z the z value to use for the point
         * \param vp_size the size of viewport
         * \return the point in world space
         */
        NODISCARD auto viewport_to_world(glm::vec2 screen, glm::ivec2 vp_size, float z = -1.0f) const noexcept -> glm::vec3;

        /**
         * \brief Converts a point in screen space to a ray in world space
         * \param screen the point in screen space; top left is (0,0), bottom right is (width, height)
         * \param vp_size the size of viewport
         * \return the ray in world space
        */
        NODISCARD auto viewport_to_ray(glm::vec2 screen, glm::ivec2 vp_size) const noexcept -> Ray;

        void set_fov(float fov) noexcept;
        void set_aspect(float aspect) noexcept;

        /**
         * \brief Rotates the camera pivoted at the center
         * \param delta the rotation in degrees
         * \note the rotation is applied in the order XYZ
        */
        void set_delta_rotation(glm::vec3 const& delta) noexcept;

        /**
         * \brief Moves the camera center by delta, in local space
         * \param delta the movement in local space
        */
        void set_delta_dolly(glm::vec3 const& delta) noexcept;

        /**
         * \brief Zooms the camera by delta
         * \param delta the zoom amount
         * \note delta > 0 zooms in, delta < 0 zooms out
        */
        void set_delta_zoom(float delta) noexcept;


        void set_eye(glm::vec3 const& eye) noexcept;
        void set_center(glm::vec3 const& center) noexcept;
        void set(LookAtParams const& params) noexcept;
        void on_deserialize() noexcept;

        auto operator==(Camera const& other) const noexcept -> bool;
        auto operator!=(Camera const& other) const noexcept -> bool;

    private:
        void on_view_changed() noexcept;
        void on_proj_changed() noexcept;

        static constexpr float k_near = 0.1f, k_far = 100000.0f;
        static constexpr float k_min_fov = 20.0f, k_max_fov = 120.0f;

        BEGIN_REFLECT(Camera, void);
        FIELD(float, m_fov, {},
            MSerialize{});
        FIELD(float, m_aspect, {},
            MSerialize{});
        // m_eye is the position of the camera
        // m_center is the point the camera is looking at
        // m_up is the up vector of the camera
        // m_arm_dir is the direction from m_eye to m_center
        FIELD(glm::vec3, m_eye, {},
            MSerialize{});
        FIELD(glm::vec3, m_center, {},
            MSerialize{});
        FIELD(glm::vec3, m_up, {},
            MSerialize{});
        END_REFLECT();
        glm::vec3 m_arm_dir;
        glm::mat4 m_cam_transform;
        glm::mat4 m_view;
        glm::mat4 m_projection;
        glm::mat4 m_view_proj;
        glm::mat4 m_inv_view_proj;
    };
}
