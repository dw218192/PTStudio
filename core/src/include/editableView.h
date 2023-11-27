#pragma once
#include "transform.h"
#include "reflection.h"
#include "editFlags.h"
#include <memory>

namespace PTS {
    struct ConstEditableInterface {
        virtual ~ConstEditableInterface() = default;
        virtual auto get_name() const noexcept -> std::string_view = 0;
        virtual auto get_transform() const noexcept -> Transform const& = 0;
        virtual auto get_edit_flags() const noexcept -> EditFlags = 0;
        virtual auto get_class_name() const noexcept -> std::string_view = 0;
        virtual auto get_addr() const noexcept -> void const* = 0;
    };

    struct EditableInterface : ConstEditableInterface {
        virtual auto set_name(std::string_view name) noexcept -> void = 0;
        virtual auto set_transform(Transform transform) noexcept -> void = 0;
        virtual auto set_edit_flags(int flags) noexcept -> void = 0;
        virtual auto get_addr() noexcept -> void* = 0;
    };
    /**
     * @brief Type-erased interface for editable objects.
     * @details This interface is used to allow the editor to modify objects without knowing their type.\n
     * However, the type needs to be reflectable as the interface provides functionalities for un-erasing
     * the type.
     * @note This interface does not store the object, nor does it own it. It has no way of knowing if the
     * object is still alive. It is the user's responsibility to ensure the object is still valid.
    */
    struct EditableView {
        friend struct ConstEditableView;
    private:
        template <typename T>
        struct EditableImpl final : EditableInterface {
            static_assert(is_reflectable<T>::value, "T must be reflectable");

            DEFAULT_COPY_MOVE(EditableImpl);

            ~EditableImpl() override = default;

            template <typename = std::enable_if_t<is_reflectable<T>::value>>
            EditableImpl(T& obj) noexcept : m_obj{ std::ref(obj) } {}

            auto get_name() const noexcept -> std::string_view override {
                return m_obj.get().get_name();
            }
            auto set_name(std::string_view name) noexcept -> void override {
                m_obj.get().set_name(name);
            }
            auto get_transform() const noexcept -> Transform const& override {
                return m_obj.get().get_transform();
            }
            auto get_edit_flags() const noexcept -> EditFlags override {
                return m_obj.get().get_edit_flags();
            }
            auto set_edit_flags(int flags) noexcept -> void override {
                m_obj.get().set_edit_flags(flags);
            }
            auto set_transform(Transform transform) noexcept -> void override {
                m_obj.get().set_transform(std::move(transform));
            }
            auto get_class_name() const noexcept -> std::string_view override {
                return T::get_class_name();
            }
            auto get_addr() noexcept -> void* override {
                return &m_obj.get();
            }
            auto get_addr() const noexcept -> void const* override {
                return &m_obj.get();
            }
        private:
            Ref<T> m_obj;
        };
    public:

        DEFAULT_COPY_MOVE(EditableView);

        virtual ~EditableView() = default;

        template <typename T, typename = std::enable_if_t<is_reflectable<T>::value>>
        EditableView(T& obj) noexcept : m_impl{ new EditableImpl { obj } } {}


        NODISCARD auto get_name() const noexcept -> std::string_view {
            return m_impl->get_name();
        }
        NODISCARD auto set_name(std::string_view name) const noexcept -> void {
            m_impl->set_name(name);
        }
        NODISCARD auto get_transform() const noexcept -> Transform const& {
            return m_impl->get_transform();
        }
        NODISCARD auto set_transform(Transform transform) const noexcept -> void {
            m_impl->set_transform(transform);
        }
        NODISCARD auto get_addr() const noexcept -> void const* {
            return m_impl->get_addr();
        }
        NODISCARD auto get_addr() noexcept -> void* {
            return m_impl->get_addr();
        }

        template<typename T>
        NODISCARD auto as() -> T* {
            if (m_impl->get_class_name() == T::get_class_name()) {
                return reinterpret_cast<T*>(get_addr());
            }
            else {
                return nullptr;
            }
        }
        template<typename T>
        NODISCARD auto as() const -> T const* {
            if (m_impl->get_class_name() == T::get_class_name()) {
                return reinterpret_cast<T const*>(get_addr());
            }
            else {
                return nullptr;
            }
        }
    private:
        std::shared_ptr<EditableInterface> m_impl;
    };

    struct ConstEditableView {
    private:
        template <typename T>
        struct ConstEditableImpl final : ConstEditableInterface {
            static_assert(is_reflectable<T>::value, "T must be reflectable");

            DEFAULT_COPY_MOVE(ConstEditableImpl);

            ~ConstEditableImpl() override = default;

            template <typename = std::enable_if_t<is_reflectable<T>::value>>
            ConstEditableImpl(T const& obj) noexcept : m_obj{ std::cref(obj) } {}

            auto get_name() const noexcept -> std::string_view override {
                return m_obj.get().get_name();
            }
            auto get_transform() const noexcept -> Transform const& override {
                return m_obj.get().get_transform();
            }
            auto get_edit_flags() const noexcept -> EditFlags override {
                return m_obj.get().get_edit_flags();
            }
            auto get_class_name() const noexcept -> std::string_view override {
                return T::get_class_name();
            }
            auto get_addr() const noexcept -> void const* override {
                return &m_obj.get();
            }
        private:
            View<T> m_obj;
        };
    public:
        DEFAULT_COPY_MOVE(ConstEditableView);
        virtual ~ConstEditableView() = default;

        template <typename T, typename = std::enable_if_t<is_reflectable<T>::value>>
        ConstEditableView(T const& obj) noexcept : m_impl{ new ConstEditableImpl { obj } } {}

        ConstEditableView(EditableView other) noexcept {
            m_impl = std::dynamic_pointer_cast<ConstEditableInterface>(other.m_impl);
        }
        auto operator=(EditableView other) noexcept -> ConstEditableView& {
            m_impl = std::dynamic_pointer_cast<ConstEditableInterface>(other.m_impl);
            return *this;
        }

        NODISCARD auto get_name() const noexcept -> std::string_view {
            return m_impl->get_name();
        }
        NODISCARD auto get_transform() const noexcept -> Transform const& {
            return m_impl->get_transform();
        }
        NODISCARD auto get_addr() const noexcept -> void const* {
            return m_impl->get_addr();
        }

        template<typename T>
        NODISCARD auto as() const -> T const* {
            if (m_impl->get_class_name() == T::get_class_name()) {
                return reinterpret_cast<T const*>(get_addr());
            }
            else {
                return nullptr;
            }
        }
    private:
        std::shared_ptr<ConstEditableInterface> m_impl;
    };

    template<typename SomeView, typename SomeView2, typename =
        std::enable_if_t<
        std::conjunction_v<
        std::disjunction<
        std::is_same<SomeView, EditableView>,
        std::is_same<SomeView, ConstEditableView>
        >,
        std::disjunction<
        std::is_same<SomeView2, EditableView>,
        std::is_same<SomeView2, ConstEditableView>
        >
        >
        >
    >
    NODISCARD auto operator==(SomeView a, SomeView2 b) noexcept -> bool {
        return a.get_addr() == b.get_addr();
    }

    template<typename SomeView, typename SomeView2, typename =
        std::enable_if_t<
        std::conjunction_v<
        std::disjunction<
        std::is_same<SomeView, EditableView>,
        std::is_same<SomeView, ConstEditableView>
        >,
        std::disjunction<
        std::is_same<SomeView2, EditableView>,
        std::is_same<SomeView2, ConstEditableView>
        >
        >
        >
    >
    NODISCARD auto operator!=(SomeView a, SomeView2 b) noexcept -> bool {
        return !(a == b);
    }
}