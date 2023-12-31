#include "sceneObject.h"
#include "camera.h"
#include "scene.h"

PTS::SceneObject::SceneObject(ObjectConstructorUsage usage) noexcept
	: Object{usage} {}

PTS::SceneObject::SceneObject(Scene& scene, std::string_view name, Transform transform, EditFlags edit_flags)
	: Object{name}, m_local_transform{std::move(transform)}, m_flags{edit_flags}, m_scene{&scene} {
	m_world_transform = m_local_transform;
}

PTS::SceneObject::SceneObject(Scene& scene, Transform transform, EditFlags edit_flags)
	: Object{scene.next_obj_name()}, m_local_transform{std::move(transform)}, m_flags{edit_flags}, m_scene{&scene} {
	m_world_transform = m_local_transform;
}

auto PTS::SceneObject::has_child(View<SceneObject> child) noexcept -> bool {
	return find_child(child) != m_children.end();
}

auto PTS::SceneObject::add_child(Ref<SceneObject> child) noexcept -> void {
	if (m_scene->is_valid_obj(child.get())) {
		auto& ch = child.get();
		if (ch.get_parent() == this) {
			return;
		}

		m_children.emplace_back(&ch);
		ch.m_parent = this;
	}
}

auto PTS::SceneObject::remove_child(Ref<SceneObject> child) noexcept -> void {
	if (m_scene->is_valid_obj(child)) {
		if (auto const it = find_child(child); it != m_children.end()) {
			child.get().m_parent = nullptr;
			m_children.erase(it);
		}
	}
}

auto PTS::SceneObject::get_transform(TransformSpace space) const noexcept -> Transform const& {
	if (space == TransformSpace::LOCAL) {
		return m_local_transform;
	} else {
		return m_world_transform;
	}
}

auto PTS::SceneObject::set_transform(Transform transform, TransformSpace space) noexcept -> void {
	if (space == TransformSpace::LOCAL) {
		m_local_transform = std::move(transform);
		if (auto const par = get_parent()) {
			m_world_transform = par->get_transform(TransformSpace::WORLD) * m_local_transform;
		} else {
			m_world_transform = m_local_transform;
		}
	} else {
		m_world_transform = std::move(transform);
		if (auto const par = get_parent()) {
			m_local_transform = par->get_transform(TransformSpace::WORLD).inverse() * m_world_transform;
		} else {
			m_local_transform = m_world_transform;
		}
	}
	get_field_info<&SceneObject::m_world_transform>().on_change(m_world_transform, *this);
	get_field_info<&SceneObject::m_local_transform>().on_change(m_local_transform, *this);

	for (auto const& ch : m_children) {
		ch->update_transform();
	}
}

auto PTS::SceneObject::set_parent(ObserverPtr<SceneObject> parent) noexcept -> void {
	if (!parent) {
		if (get_parent()) {
			m_parent->remove_child(*this);
		}
		m_parent = parent;
		get_field_info<&SceneObject::m_parent>().on_change(m_parent, *this);
	} else {
		if (m_scene->is_valid_obj(*parent)) {
			if (!parent->has_child(*this)) {
				if (get_parent()) {
					m_parent->remove_child(*this);
				}
				parent->add_child(*this);
				m_parent = parent;
				get_field_info<&SceneObject::m_parent>().on_change(m_parent, *this);
			}
		}
	}
}

auto PTS::SceneObject::get_parent() const noexcept -> ObserverPtr<SceneObject> {
	if (m_parent) {
		return m_scene->is_valid_obj(*m_parent) ? m_parent : nullptr;
	} else {
		return nullptr;
	}
}

auto PTS::SceneObject::set_edit_flags(EditFlags flags) noexcept -> void {
	auto const prev_flags = m_flags;
	m_flags = flags;
	if (!(prev_flags & EditFlags::_NoEdit) && (m_flags & EditFlags::_NoEdit)) {
		get_scene()->try_remove_editable(*this);
	}
	get_field_info<&SceneObject::m_flags>().on_change(m_flags, *this);
}

auto PTS::SceneObject::find_child(
	View<SceneObject> child) const -> std::vector<ObserverPtr<SceneObject>>::const_iterator {
	if (!m_scene->is_valid_obj(child.get())) {
		return m_children.end();
	}
	return std::find(m_children.begin(), m_children.end(), &child.get());
}

auto PTS::SceneObject::update_transform() noexcept -> void {
	if (auto const par = get_parent()) {
		m_world_transform = par->get_transform(TransformSpace::WORLD) * m_local_transform;
		get_field_info<&SceneObject::m_world_transform>().on_change(m_world_transform, *this);

		for (auto const& ch : m_children) {
			ch->update_transform();
		}
	}
}
