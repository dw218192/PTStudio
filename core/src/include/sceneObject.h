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
		SceneObject(Scene& scene, std::string_view name, Transform transform, EditFlags edit_flags);
		SceneObject(Scene& scene, Transform transform, EditFlags edit_flags);

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

		NODISCARD auto is_editable() const noexcept -> bool {
			return !(m_flags & _NoEdit);
		}

		auto has_child(View<SceneObject> child) noexcept -> bool;
		auto add_child(Ref<SceneObject> child) noexcept -> void;
		auto remove_child(Ref<SceneObject> child) noexcept -> void;
		auto set_transform(Transform transform, TransformSpace space) noexcept -> void;
		auto set_parent(ObserverPtr<SceneObject> parent) noexcept -> void;
		auto get_parent() const noexcept -> ObserverPtr<SceneObject>;

		auto set_edit_flags(EditFlags flags) noexcept -> void;

		struct FieldTag {
			static constexpr int WORLD_TRANSFORM = 0;
			static constexpr int LOCAL_TRANSFORM = 1;
			static constexpr int FLAGS = 2;
			static constexpr int SCENE = 3;
			static constexpr int PARENT = 4;
			static constexpr int CHILDREN = 5;
		};

	private:
		auto find_child(View<SceneObject> child) const -> std::vector<ObserverPtr<SceneObject>>::const_iterator;
		auto update_transform() noexcept -> void;

		BEGIN_REFLECT(SceneObject, Object);
		FIELD(Transform, m_world_transform, {},
		      MSerialize{}, MNoInspect{}); // handled explicitly
		FIELD(Transform, m_local_transform, {},
		      MSerialize{}, MNoInspect{}); // handled explicitly

		FIELD(EditFlags, m_flags, k_editable_flags,
		      MSerialize{}, k_edit_flags_modifier);

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
