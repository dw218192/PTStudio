#pragma once

#include "ext.h"

enum class TransformSpace {
	LOCAL, GLOBAL
};

struct Transform {
    // creates an identity transform
    Transform() noexcept;
    // create a transform from a position, rotation, and scale
    // rotation is in degrees, and is applied in the order XYZ
    Transform(glm::vec3 const& pos, glm::vec3 const& rot, glm::vec3 const& scale = glm::vec3(1,1,1)) noexcept;
    static auto look_at(glm::vec3 const& pos, glm::vec3 const& target, glm::vec3 const& up) noexcept -> Transform;

    [[nodiscard]] auto get_position() const noexcept -> glm::vec3 const& { return m_pos; }
    [[nodiscard]] auto get_rotation() const noexcept -> glm::vec3 const& { return m_rot; }
    [[nodiscard]] auto get_scale() const noexcept -> glm::vec3 const& { return m_scale; }
    [[nodiscard]] auto get_matrix() const noexcept -> glm::mat4 const& { return m_trans; }

    void set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept;
    void set_position(TransformSpace space, glm::vec3 const& pos) noexcept;
    void set_scale(TransformSpace space, glm::vec3 const& scale) noexcept;

private:
    glm::vec3 m_pos;
    glm::vec3 m_rot;
    glm::vec3 m_scale;
    glm::mat4 m_trans;
};