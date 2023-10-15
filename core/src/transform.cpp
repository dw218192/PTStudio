#include "include/transform.h"

#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

static void decompose(glm::mat4 const& mat, glm::vec3* ppos, glm::vec3* prot, glm::vec3* pscale) {
    if (ppos) {
        *ppos = glm::column(mat, 3);
    }

    if(prot || pscale) {
        glm::vec3 r1 = glm::row(mat, 0),
            r2 = glm::row(mat, 1),
            r3 = glm::row(mat, 2);

    	glm::vec3 const scale = glm::vec3{
            glm::length(r1),
            glm::length(r2),
            glm::length(r3)
        };
        assert(glm::length(r1) > 0.0 && glm::length(r2) > 0.0 && glm::length(r3) > 0.0);

        if (pscale) {
            *pscale = scale;
        }
        if (prot) {
            r1 /= scale.x;
            r2 /= scale.y;
            r3 /= scale.z;
            *prot = glm::vec3{
                glm::degrees(atan2(r3.y, r3.z)),
                glm::degrees(atan2(-r3.x, sqrt(r3.y * r3.y + r3.z * r3.z))),
                glm::degrees(atan2(r2.x, r1.x))
            };
        }
    }
}

static glm::mat4 compose(glm::vec3 const& pos, glm::vec3 const& rot, glm::vec3 const& scale) {
    glm::mat4 const t = glm::translate(pos);
    glm::mat4 const r = glm::eulerAngleXYZ(glm::radians(rot.x), glm::radians(rot.y), glm::radians(rot.z));
    glm::mat4 const s = glm::scale(scale);
    return t * r * s;
}

Transform::Transform() noexcept 
    : m_pos{0,0,0}, m_rot{0,0,0}, m_scale{1,1,1}, m_trans(1.0), m_inv_trans(1.0)
{ }

Transform::Transform(glm::vec3 const& pos, glm::vec3 const& rot, glm::vec3 const& scale) noexcept
    : m_pos{ pos }, m_rot{ rot }, m_scale{ scale }
{
    m_trans = compose(m_pos, m_rot, m_scale);
    m_inv_trans = glm::inverse(m_trans);
}

Transform::Transform(glm::mat4 const& matrix) noexcept : m_trans{matrix}  {
    update_matrix();
}

auto Transform::look_at(glm::vec3 const& pos, glm::vec3 const& target, glm::vec3 const& up) noexcept -> Transform {
    Transform ret;
    ret.m_trans = glm::lookAt(pos, target, up);
    decompose(ret.m_trans, &ret.m_pos, &ret.m_rot, &ret.m_scale);
    return ret;
}

auto Transform::inverse() const noexcept -> Transform {
    return Transform{ glm::inverse(get_matrix()) };
}

void Transform::multiply(TransformSpace space, glm::mat4 const& transform) noexcept {
    if (space == TransformSpace::WORLD) {
        m_trans = transform * m_trans;
    } else {
        m_trans *= transform;
    }
}

void Transform::set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept {
    multiply(space, glm::eulerAngleXYZ(glm::radians(rot.x), glm::radians(rot.y), glm::radians(rot.z)));
    update_matrix();
}

void Transform::set_position(TransformSpace space, glm::vec3 const& pos) noexcept {
    multiply(space, glm::translate(pos));
    update_matrix();
}

void Transform::set_scale(TransformSpace space, glm::vec3 const& scale) noexcept {
    multiply(space, glm::scale(scale));
    update_matrix();
}

void Transform::update_matrix() noexcept {
    m_inv_trans = glm::inverse(m_trans);
    decompose(m_trans, &m_pos, &m_rot, &m_scale);
}

auto Transform::local_to_world_pos(glm::vec3 local) const noexcept -> glm::vec3 {
    return m_trans * glm::vec4(local, 1.0f);
}

auto Transform::world_to_local_pos(glm::vec3 world) const noexcept -> glm::vec3 {
    return m_inv_trans * glm::vec4(world, 1.0f);
}

auto Transform::local_to_world_dir(glm::vec3 local) const noexcept -> glm::vec3 {
    return m_trans * glm::vec4(local, 0.0f);
}

auto Transform::world_to_local_dir(glm::vec3 world) const noexcept -> glm::vec3 {
    return m_inv_trans * glm::vec4(world, 0.0f);
}

auto Transform::local_to_world_len(float len) const noexcept -> float {
    return len * glm::length(m_scale);
}

auto Transform::world_to_local_len(float len) const noexcept -> float {
    return len / glm::length(m_scale);
}