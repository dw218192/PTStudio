#pragma once

#include <tl/expected.hpp>
#include <string>
#include <optional>

#include "transform.h"
#include "object.h"
#include "reflection.h"
#include "editFlags.h"

namespace PTS {
	struct Scene;

	struct SceneObject : Object {
		SceneObject(ObjectConstructorUsage usage = ObjectConstructorUsage::SERIALIZE) noexcept;
		SceneObject(Scene& scene, std::string_view name, Transform transform);
		SceneObject(Scene& scene, Transform transform);

		NODISCARD auto get_scene() const noexcept -> ObserverPtr<Scene> {
			return m_scene;
		}

		/**
		 * @brief Gets the transform of this object in the given space.
		 * @note If the space is LOCAL, the transform returned transforms from the object's local space
		 * to its parent's local space.
		 * @note If the space is WORLD, the transform returned transforms from the object's local space
		 * to world space.
		 * @param space The space to get the transform in.
		 * @return The transform of this object in the given space.
		*/
		NODISCARD auto get_transform(TransformSpace space) const noexcept -> Transform const&;
		NODISCARD auto get_edit_flags() const noexcept {
			return m_flags;
		}

		auto has_child(View<SceneObject> child) noexcept -> bool;
		auto add_child(Ref<SceneObject> child) noexcept -> void;
		auto remove_child(Ref<SceneObject> child) noexcept -> void;
		auto set_transform(Transform transform, TransformSpace space) noexcept -> void;
		auto set_parent(ObserverPtr<SceneObject> parent) noexcept -> void;
		auto get_parent() const noexcept -> ObserverPtr<SceneObject>;

		auto set_edit_flags(int flags) noexcept -> void;

	private:
		auto find_child(View<SceneObject> child) const -> std::vector<ObserverPtr<SceneObject>>::const_iterator;
		auto update_transform() noexcept -> void;

		DECL_DEFERRED_STATIC_INIT(static_init);

		BEGIN_REFLECT(SceneObject, Object);
		FIELD(Transform, m_world_transform, {},
		      MSerialize{}, MNoInspect{}); // handled explicitly
		FIELD(Transform, m_local_transform, {},
		      MSerialize{}, MNoInspect{}); // handled explicitly

		FIELD(EditFlags, m_flags, static_cast<EditFlags>(EditFlags::Visible | EditFlags::Selectable),
		      MSerialize{}, edit_flags_modifier);

		FIELD(ObserverPtr<Scene>, m_scene, nullptr,
		      MSerialize{}, MReadOnly{}); // not editable
		FIELD(ObserverPtr<SceneObject>, m_parent, nullptr,
		      MSerialize{}, MReadOnly{}); // not editable
		FIELD(std::vector<ObserverPtr<SceneObject>>, m_children, {},
		      MSerialize{}, MReadOnly{}); // not editable

		END_REFLECT();
		// enables dynamic retrieval of class info for polymorphic types
		DECL_DYNAMIC_INFO();
	};
}
