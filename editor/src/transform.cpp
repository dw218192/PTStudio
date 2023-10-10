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
    : m_pos{0,0,0}, m_rot{0,0,0}, m_scale{1,1,1}, m_trans(1.0)
{ }

Transform::Transform(glm::vec3 const& pos, glm::vec3 const& rot, glm::vec3 const& scale) noexcept
    : m_pos{ pos }, m_rot{ rot }, m_scale{ scale }
{
    m_trans = compose(pos, rot, scale);
}

auto Transform::look_at(glm::vec3 const& pos, glm::vec3 const& target, glm::vec3 const& up) noexcept -> Transform {
    Transform ret;
    ret.m_trans = glm::lookAt(pos, target, up);
    decompose(ret.m_trans, &ret.m_pos, &ret.m_rot, &ret.m_scale);
    return ret;
}

void Transform::set_rotation(TransformSpace space, glm::vec3 const& rot) noexcept {
    if(space == TransformSpace::GLOBAL) {
        m_rot = rot;
    } else {
        m_rot += rot;
    }
    m_trans = compose(m_pos, m_rot, m_scale);
}

void Transform::set_position(TransformSpace space, glm::vec3 const& pos) noexcept {
    if (space == TransformSpace::GLOBAL) {
        m_pos = pos;
    } else {
        m_pos += pos * m_scale;
    }
    m_trans = compose(m_pos, m_rot, m_scale);
}

void Transform::set_scale(TransformSpace space, glm::vec3 const& scale) noexcept {
    if (space == TransformSpace::GLOBAL) {
        m_scale = scale;
    } else {
        m_scale *= scale;
    }
    m_trans = compose(m_pos, m_rot, m_scale);
}