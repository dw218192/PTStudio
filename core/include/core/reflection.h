#pragma once
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <variant>

// data structures
#include <array>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "callbackList.h"
#include "stringManip.h"
#include "typeTraitsUtil.h"

namespace PTS {
#pragma region Modifiers
/**
 * simple reflection framework
 * limitations:
 *  - only single declaration per line
 *  - FIELD(type<a,b,c,..>, name) doesn't work, alias the type first
 *  - relies on non-standard __COUNTER__ macro, but g++, clang, and msvc all
 * support it
 *  - access control change in END_REFLECT() may unintentionally expose private
 * members
 */

/**
 * a modifier is an attribute that can be attached to a field
 * modifiers are used to change the behavior of the inspector and the
 * serialization process modifiers are optional
 */
template <typename... Modifiers>
struct ModifierPack {
    std::tuple<Modifiers...> args;

    constexpr ModifierPack(Modifiers... args) : args(std::make_tuple(args...)) {
    }

    // not marked as constexpr because void* conversion is prohibited in
    // constexpr
    template <typename Mod>
    constexpr auto get() const -> Mod const* {
        constexpr auto index = Traits::find_v<decltype(args), Mod>;
        if constexpr (index != static_cast<std::size_t>(-1)) {
            return &std::get<index>(args);
        } else {
            return nullptr;
        }
    }

    template <typename Mod>
    constexpr auto has() const {
        return Traits::contains_v<decltype(args), Mod>;
    }
};

/**
 * @brief displays the field as a slider, only works for float and int
 * @tparam T the type of the field
 */
template <typename T>
struct MRange {
    T min, max;
    static constexpr std::string_view name = "range";
    constexpr MRange(T min, T max) : min(min), max(max) {
    }
};

/**
 * @brief mark the field as serializable, only fields with this modifier will be
 * saved to the scene file
 */
struct MSerialize {
    static constexpr std::string_view name = "serialize";
};

/**
 * @brief don't show this field in the inspector, can be used to hide explicitly
 * handled fields or fields that shouldn't be edited
 */
struct MNoInspect {
    static constexpr std::string_view name = "no inspect";
};

/**
 * @brief mark the field as read only, can be used to prevent editing of fields
 * that are calculated from other fields or for debugging purposes
 */
struct MReadOnly {
    static constexpr std::string_view name = "read only";
};

/**
 * @brief displays a color picker, only works for glm::vec3 and glm::vec4
 */
struct MColor {
    static constexpr std::string_view name = "color";
};

/**
 * @brief displays an enum as a dropdown
 */
struct MEnum {
    static constexpr std::string_view name = "enum";
    int num_items;
    auto (*get_name)(int idx) -> char const*;
};

/**
 * @brief displays a multi-select enum as a dropdown
 */
struct MEnumFlags {
    static constexpr std::string_view name = "enum flags";
    int num_items;
    auto (*get_name)(int idx) -> char const*;
};

#pragma endregion Modifiers

#pragma region TField
template <typename FieldInfoType, typename ClassType, typename FieldType>
struct OnChangeCallbackData {
    FieldInfoType const& field_info;
    FieldType const& new_val;
    ClassType& obj;
};

/**
 * @brief proxy for a field instance, used to intercept field access and
 * modification
 */
template <typename T, typename FieldInfoType>
struct FieldProxy {
    using value_type = T;
    using field_info_type = FieldInfoType;
    using class_type = typename FieldInfoType::class_type;

    template <typename U>
    FieldProxy(U& ref, field_info_type& field_info, class_type& obj,
               std::enable_if_t<!std::is_same_v<std::remove_cv_t<U>, FieldProxy>>* = nullptr)
        : m_ref{ref}, m_field_info{field_info}, m_obj{obj} {
    }

    FieldProxy(FieldProxy&& other) = default;
    FieldProxy(FieldProxy const& other) = default;
    auto operator=(FieldProxy&& other) -> FieldProxy& = default;
    auto operator=(FieldProxy const& other) -> FieldProxy& = default;
    ~FieldProxy() = default;

    template <typename U>
    auto operator=(U&& new_val) -> std::enable_if_t<
        std::conjunction_v<
            std::negation<std::is_same<std::remove_cv_t<std::remove_reference_t<U>>, FieldProxy>>,
            std::negation<std::is_const<T>>, std::is_assignable<T&, U&&>>,
        FieldProxy&> {
        m_ref.get() = std::forward<U>(new_val);
        m_field_info.get().on_change(m_ref.get(), m_obj.get());
        return *this;
    }

    template <typename U>
    auto operator=(std::initializer_list<U> new_val)
        -> std::enable_if_t<std::conjunction_v<std::negation<std::is_const<T>>,
                                               std::is_assignable<T&, std::initializer_list<U>>>,
                            FieldProxy&> {
        m_ref.get() = new_val;
        m_field_info.get().on_change(m_ref.get(), m_obj.get());
        return *this;
    }

    // arithmetic operators
#define DEFINE_ASSIGN_OP(op)                                                             \
    template <typename U,                                                                \
              typename = std::void_t<decltype(std::declval<T>() op std::declval<U>())>>  \
    auto operator op##=(U&& other)->std::enable_if_t<!std::is_const_v<T>, FieldProxy&> { \
        decltype(auto) res{m_ref.get() op std::forward<U>(other)};                       \
        *this = res;                                                                     \
        return *this;                                                                    \
    }
    DEFINE_ASSIGN_OP(+)
    DEFINE_ASSIGN_OP(-)
    DEFINE_ASSIGN_OP(*)
    DEFINE_ASSIGN_OP(/)
    DEFINE_ASSIGN_OP(%)
    DEFINE_ASSIGN_OP(&)
    DEFINE_ASSIGN_OP(|)
    DEFINE_ASSIGN_OP(^)
    DEFINE_ASSIGN_OP(<<)
    DEFINE_ASSIGN_OP(>>)

#undef DEFINE_ASSIGN_OP

    auto operator->() -> std::enable_if_t<!std::is_const_v<T>, T*> {
        return &m_ref.get();
    }

    auto operator->() const -> const T* {
        return &m_ref.get();
    }

    auto operator*() -> std::enable_if_t<!std::is_const_v<T>, T&> {
        return m_ref.get();
    }

    auto operator*() const -> const T& {
        return m_ref.get();
    }

    template <auto MemFunc, typename... Args>
    auto invoke(Args&&... args) -> decltype(auto) {
        if constexpr (std::is_const_v<T>) {
            return std::invoke(MemFunc, m_ref.get(), std::forward<Args>(args)...);
        } else {
            if constexpr (std::is_void_v<decltype(std::invoke(MemFunc, std::declval<T>(),
                                                              std::declval<Args>()...))>) {
                std::invoke(MemFunc, m_ref.get(), std::forward<Args>(args)...);
                if constexpr (!Traits::is_const_callable_method_v<MemFunc, T, Args...>) {
                    m_field_info.get().on_change(m_ref.get(), m_obj.get());
                }
            } else {
                decltype(auto) ret{std::invoke(MemFunc, m_ref.get(), std::forward<Args>(args)...)};
                if constexpr (!Traits::is_const_callable_method_v<MemFunc, T, Args...>) {
                    m_field_info.get().on_change(m_ref.get(), m_obj.get());
                }
                return ret;
            }
        }
    }

    auto get() const -> T const& {
        return m_ref.get();
    }

    auto get() -> std::enable_if_t<!std::is_const_v<T>, T&> {
        return m_ref.get();
    }

   private:
    std::reference_wrapper<value_type> m_ref;
    std::reference_wrapper<field_info_type> m_field_info;
    std::reference_wrapper<class_type> m_obj;
};

/**
 * @brief non-constexpr part of TField
 */
template <typename ClassType, typename FieldType, typename ModifierPackType>
struct TFieldRuntime;

/**
 * @brief templated field info, used as metadata for a field
 */
template <typename ClassType, typename FieldType, typename ModifierPackType = ModifierPack<>>
struct TField {
    using type = FieldType;
    using class_type = ClassType;
    using modifier_pack_type = ModifierPackType;
    using callback_type =
        typename TFieldRuntime<ClassType, FieldType, ModifierPackType>::callback_type;
    using callback_data_type =
        typename TFieldRuntime<ClassType, FieldType, ModifierPackType>::callback_data_type;

    constexpr TField(int member_index, std::string_view var_name, std::string_view type_name,
                     TFieldRuntime<ClassType, FieldType, ModifierPackType>& runtime_info,
                     ModifierPackType modifier)
        : member_index(member_index),
          var_name(var_name),
          type_name(type_name),
          runtime_info(runtime_info),
          modifiers(modifier) {
    }

    int member_index;
    std::string_view var_name;
    std::string_view type_name;
    TFieldRuntime<ClassType, FieldType, ModifierPackType>& runtime_info;
    ModifierPackType modifiers;

    template <typename Mod>
    constexpr auto get_modifier() const -> Mod const* {
        return modifiers.template get<Mod>();
    }

    template <typename Mod>
    constexpr auto has_modifier() const {
        return modifiers.template has<Mod>();
    }

    // -----------------  runtime only -----------------

    /**
     * @brief get the value of the field given an instance of the class
     * @param obj the instance of the class
     * @return the value of the field
     */
    auto get(ClassType const& obj) const -> auto const& {
        return obj.*(runtime_info.mem_pointer);
    }

    /**
     * @brief get the value of the field given an instance of the class
     * @param obj the instance of the class
     * @return the value of the field
     */
    auto get(ClassType& obj) -> auto& {
        return obj.*(runtime_info.mem_pointer);
    }

    /**
     * @brief get a proxy for the field given an instance of the class,
     * on_change callbacks will be called when the proxy is modified
     * @param obj the instance of the class
     * @return a proxy for the field
     */
    auto get_proxy(ClassType const& obj) const {
        return FieldProxy<FieldType const, TField const>{obj.*(runtime_info.mem_pointer), *this,
                                                         obj};
    }

    /**
     * @brief get a proxy for the field given an instance of the class,
     * on_change callbacks will be called when the proxy is modified
     * @param obj the instance of the class
     * @return a proxy for the field
     */
    auto get_proxy(ClassType& obj) const {
        return FieldProxy<FieldType, TField const>{obj.*(runtime_info.mem_pointer), *this, obj};
    }

    /**
     * @brief returns the default value of the field specified in the in-class
     * field initializer
     * @return the default value of the field
     */
    auto get_default() const -> FieldType {
        return runtime_info.default_val;
    }

    auto get_on_change_callback_list() const -> auto& {
        return runtime_info.on_change_callbacks;
    }

    /**
     * @brief calls all registered on_change callbacks
     * @param new_val the new value of the field
     * @param obj the instance of the class
     * @return void
     */
    auto on_change(FieldType const& new_val, ClassType& obj) const -> void {
        runtime_info.on_change_callbacks(callback_data_type{*this, new_val, obj});
    }
};

template <typename ClassType, typename FieldType, typename ModifierPackType>
struct TFieldRuntime {
    friend struct TField<ClassType, FieldType, ModifierPackType>;
    using callback_data_type =
        OnChangeCallbackData<TField<ClassType, FieldType, ModifierPackType>, ClassType, FieldType>;
    using callback_type = void(callback_data_type);

    TFieldRuntime(int member_index, FieldType default_val, FieldType ClassType::* mem_pointer)
        : member_index(member_index),
          default_val(std::move(default_val)),
          mem_pointer(mem_pointer) {
    }

   private:
    int member_index;
    FieldType default_val;
    FieldType ClassType::* mem_pointer;
    CallbackList<callback_type> on_change_callbacks;
};

#pragma endregion TField

namespace Traits {
// loose interface for TField
template <typename T, typename = void>
struct has_tfield_interface : std::false_type {};

template <template <typename, typename, typename...> typename T, typename ClassType,
          typename FieldType, typename... ExtraTypes>
struct has_tfield_interface<
    T<ClassType, FieldType, ExtraTypes...>,
    std::void_t<typename T<ClassType, FieldType, ExtraTypes...>::type,
                decltype(std::declval<T<ClassType, FieldType, ExtraTypes...>>().get(
                    std::declval<ClassType>())),
                decltype(std::declval<T<ClassType, FieldType, ExtraTypes...>>().var_name),
                decltype(std::declval<T<ClassType, FieldType, ExtraTypes...>>().type_name)>>
    : std::enable_if_t<
          std::conjunction_v<
              std::is_same<FieldType, typename T<ClassType, FieldType, ExtraTypes...>::type>,
              std::is_same<FieldType, std::decay_t<decltype(std::declval<T<ClassType, FieldType,
                                                                           ExtraTypes...>>()
                                                                .get(std::declval<ClassType>()))>>,
              std::is_same<
                  std::string_view,
                  decltype(std::declval<T<ClassType, FieldType, ExtraTypes...>>().var_name)>,
              std::is_same<
                  std::string_view,
                  decltype(std::declval<T<ClassType, FieldType, ExtraTypes...>>().type_name)>>,
          std::true_type> {};

template <typename T>
constexpr bool has_tfield_interface_v = has_tfield_interface<T>::value;

template <typename T, typename F, typename = void>
struct get_for_each_field_impl {
    static constexpr auto do_it(F&& func) {
    }
};

template <typename T, typename F>
struct get_for_each_field_impl<T, F, std::enable_if_t<is_reflectable_v<T>>> {
    static constexpr auto do_it(F&& func) {
        T::for_each_field(std::forward<F>(func));
    }
};

// type to name registeration
template <typename T, typename = void>
struct type_to_name {
    static constexpr std::string_view value = "unknown";
};

template <typename T>
struct type_to_name<T, std::enable_if_t<is_reflectable_v<T>>> {
    static constexpr std::string_view value = T::get_class_name();
};

template <typename T>
constexpr auto type_to_name_v = type_to_name<T>::value;

// register common types
#define DECL_TYPE_TO_NAME(_type)                          \
    template <>                                           \
    struct type_to_name<_type> {                          \
        static constexpr std::string_view value = #_type; \
    }
// for template, we only care about the outer type name
#define DECL_TYPE_TO_NAME_TEMPLATE(_type)                 \
    template <typename... Args>                           \
    struct type_to_name<_type<Args...>> {                 \
        static constexpr std::string_view value = #_type; \
    }

DECL_TYPE_TO_NAME(int);

DECL_TYPE_TO_NAME(float);

DECL_TYPE_TO_NAME(double);

DECL_TYPE_TO_NAME(bool);

DECL_TYPE_TO_NAME(char);

DECL_TYPE_TO_NAME(unsigned char);

DECL_TYPE_TO_NAME(short);

DECL_TYPE_TO_NAME(unsigned short);

DECL_TYPE_TO_NAME(long);

DECL_TYPE_TO_NAME(unsigned long);

DECL_TYPE_TO_NAME(long long);

DECL_TYPE_TO_NAME(unsigned long long);

DECL_TYPE_TO_NAME(void);

DECL_TYPE_TO_NAME(std::nullptr_t);

DECL_TYPE_TO_NAME(std::string);

DECL_TYPE_TO_NAME(std::string_view);

DECL_TYPE_TO_NAME_TEMPLATE(std::vector);

// std array is special because it has a non-templated argument
template <typename T, std::size_t N>
struct type_to_name<std::array<T, N>> {
    static constexpr std::string_view value = "std::array";
};

DECL_TYPE_TO_NAME_TEMPLATE(std::unordered_map);

DECL_TYPE_TO_NAME_TEMPLATE(std::unordered_set);

DECL_TYPE_TO_NAME_TEMPLATE(std::map);

DECL_TYPE_TO_NAME_TEMPLATE(std::set);

DECL_TYPE_TO_NAME_TEMPLATE(std::queue);

DECL_TYPE_TO_NAME_TEMPLATE(std::stack);

DECL_TYPE_TO_NAME_TEMPLATE(std::list);

DECL_TYPE_TO_NAME_TEMPLATE(std::deque);

// bitset is special because it has a non-templated argument
template <std::size_t N>
struct type_to_name<std::bitset<N>> {
    static constexpr std::string_view value = "std::bitset";
};

DECL_TYPE_TO_NAME_TEMPLATE(std::tuple);

DECL_TYPE_TO_NAME_TEMPLATE(std::pair);

DECL_TYPE_TO_NAME_TEMPLATE(std::optional);

DECL_TYPE_TO_NAME_TEMPLATE(std::variant);

DECL_TYPE_TO_NAME_TEMPLATE(std::function);

DECL_TYPE_TO_NAME_TEMPLATE(std::unique_ptr);

DECL_TYPE_TO_NAME_TEMPLATE(std::shared_ptr);

DECL_TYPE_TO_NAME_TEMPLATE(std::weak_ptr);

DECL_TYPE_TO_NAME_TEMPLATE(std::allocator);

DECL_TYPE_TO_NAME_TEMPLATE(std::char_traits);
}  // namespace Traits

#pragma region DynamicTypeInfo
// used to dynamically get info about a class
struct ClassInfo {
    std::string_view class_name;
    int num_members;
    ClassInfo const* parent;
};

// interface for type-erased iterators
struct TypeErasedIteratorInterface {
    virtual ~TypeErasedIteratorInterface() = default;
    virtual auto operator++() -> TypeErasedIteratorInterface& = 0;
    virtual auto operator*() const -> void const* = 0;
    virtual auto operator*() -> void* = 0;
    virtual auto operator==(TypeErasedIteratorInterface const& other) const -> bool = 0;
    virtual auto operator!=(TypeErasedIteratorInterface const& other) const -> bool = 0;
};

// type-erased iterator
struct TypeErasedIterator {
    template <typename Where>
    struct IteratorWrapper : TypeErasedIteratorInterface {
        IteratorWrapper(Where& where, size_t idx) : m_where(where), m_it(idx) {
        }

        auto operator++() -> TypeErasedIteratorInterface& override {
            ++m_it;
            return *this;
        }

        auto operator*() const -> void const* override {
            return const_cast<IteratorWrapper*>(this)->operator*();
        }

        auto operator*() -> void* override {
            if constexpr (Traits::is_sequence_container_v<Where>) {
                if (m_where.get().size() <= m_it) {
                    m_where.get().resize(m_it + 1);
                }
                return &*std::next(m_where.get().begin(), m_it);
            } else {
                return &*(&m_where.get() + m_it);
            }
        }

        auto operator==(TypeErasedIteratorInterface const& other) const -> bool override {
            if (auto* other_it = dynamic_cast<IteratorWrapper const*>(&other)) {
                return m_it == other_it->m_it && &m_where.get() == &other_it->m_where.get();
            }
            return false;
        }

        auto operator!=(TypeErasedIteratorInterface const& other) const -> bool override {
            return !(*this == other);
        }

       private:
        std::reference_wrapper<Where> m_where;
        size_t m_it{0};
    };

    template <typename Where>
    TypeErasedIterator(Where& where, size_t idx)
        : m_impl(std::make_shared<IteratorWrapper<Where>>(where, idx)) {
    }

    TypeErasedIterator() : m_impl(nullptr) {
    }

    auto operator++() const -> TypeErasedIteratorInterface& {
        return m_impl->operator++();
    }

    auto operator*() const -> void const* {
        return m_impl->operator*();
    }
    auto operator*() -> void* {
        return m_impl->operator*();
    }

    auto operator==(TypeErasedIterator const& other) const -> bool {
        return m_impl->operator==(*other.m_impl);
    }

    auto operator!=(TypeErasedIterator const& other) const -> bool {
        return m_impl->operator!=(*other.m_impl);
    }

   private:
    std::shared_ptr<TypeErasedIteratorInterface> m_impl;
};

/**
 * @brief represents a type, used to dynamically get info about a field
 * @note ignores cv qualifiers and references
 */
struct Type {
    std::size_t normalized_type_hash{0};  // no good way to get a readable name as of C++17

    bool is_pointer{false};
    bool is_function{false};
    bool is_container{false};
    bool is_templated{false};
    bool is_reflectable{false};
    bool is_arithmetic{false};

    template <typename T>
    static auto of() -> Type {
        auto ret = Type{};
        using raw_t = Traits::extract_raw_type_t<T>;
        ret.normalized_type_hash = typeid(raw_t).hash_code();

        if constexpr (std::is_pointer_v<raw_t>) {
            ret.is_pointer = true;
            ret.m_pointed_to_type = std::make_shared<Type>(of<std::remove_pointer_t<raw_t>>());
        }
        if constexpr (std::is_function_v<raw_t>) {
            ret.is_function = true;

            auto i = 0;
            Traits::for_each::type_in_tuple<Traits::get_function_params_t<raw_t>>(
                [&ret, &i](auto type) {
                    using type_t = std::remove_pointer_t<decltype(type)>;
                    if (i == 0) {
                        ret.m_func_return_type = std::make_shared<Type>(of<type_t>());
                    } else {
                        ret.m_func_arg_types.emplace_back(std::make_shared<Type>(of<type_t>()));
                    }
                    ++i;
                });
        }
        if constexpr (std::is_array_v<raw_t>) {
            ret.is_container = true;
            ret.m_contained_type =
                std::make_shared<Type>(of<std::remove_pointer_t<std::decay_t<raw_t>>>());
        }
        if constexpr (Traits::is_container_v<raw_t>) {
            ret.is_container = true;
            ret.m_contained_type = std::make_shared<Type>(of<typename raw_t::value_type>());
        }
        if constexpr (Traits::is_templated_v<raw_t>) {
            ret.is_templated = true;
            Traits::for_each::type_in_tuple<Traits::get_template_args_t<raw_t>>([&ret](auto type) {
                using type_t = std::remove_pointer_t<decltype(type)>;
                ret.m_template_args.emplace_back(std::make_shared<Type>(of<type_t>()));
            });
            ret.m_outer_type_name = Traits::type_to_name_v<raw_t>;
        }
        if constexpr (Traits::is_reflectable_v<raw_t>) {
            ret.is_reflectable = true;
            if constexpr (!std::is_same_v<typename raw_t::parent_type, void>) {
                ret.m_inherited_from_type =
                    std::make_shared<Type>(of<typename raw_t::parent_type>());
            }
        }
        if constexpr (std::is_arithmetic_v<raw_t>) {
            ret.is_arithmetic = true;
        }
        ret.m_type_name = Traits::type_to_name_v<raw_t>;

        return ret;
    }

    Type(Type const&) = default;
    Type(Type&&) = default;
    auto operator=(Type const&) -> Type& = default;
    auto operator=(Type&&) -> Type& = default;
    ~Type() = default;

    auto to_string() const -> std::string {
        if (is_pointer) {
            return m_pointed_to_type->to_string() + " *";
        } else if (is_function) {
            auto ret = m_func_return_type->to_string();
            ret.push_back('(');
            for (auto& arg : m_func_arg_types) {
                ret += arg->to_string();
                ret += ", ";
            }
            ret.pop_back();
            ret.pop_back();
            ret.push_back(')');
            return ret;
        } else if (is_templated) {
            auto ret = std::string{m_outer_type_name};
            ret.push_back('<');
            for (auto& arg : m_template_args) {
                ret += arg->to_string();
                ret += ", ";
            }
            ret.pop_back();
            ret.pop_back();
            ret.push_back('>');
            return ret;
        } else {
            return std::string{m_type_name};
        }
    }

    auto is_valid() const noexcept {
        return normalized_type_hash != 0;
    }

    auto pointed_to_type() const -> Type {
        return m_pointed_to_type ? *m_pointed_to_type : Type{};
    }

    auto inherited_from_type() const -> Type {
        return m_inherited_from_type ? *m_inherited_from_type : Type{};
    }

    auto func_return_type() const -> Type {
        return m_func_return_type ? *m_func_return_type : Type{};
    }

    auto func_arg_types() const -> std::vector<Type> {
        std::vector<Type> ret;
        ret.reserve(m_func_arg_types.size());
        for (auto& arg : m_func_arg_types) {
            ret.emplace_back(*arg);
        }
        return ret;
    }

    auto contained_type() const -> Type {
        return m_contained_type ? *m_contained_type : Type{};
    }

    auto template_args() const -> std::vector<Type> {
        std::vector<Type> ret;
        ret.reserve(m_template_args.size());
        for (auto& arg : m_template_args) {
            ret.emplace_back(*arg);
        }
        return ret;
    }

    auto has_common_base_with(Type const& other) const -> bool {
        if (!is_valid() || !other.is_valid()) return false;
        if (normalized_type_hash == other.normalized_type_hash) {
            return true;
        } else if (m_inherited_from_type) {
            return m_inherited_from_type->has_common_base_with(other) ||
                   has_common_base_with(other.inherited_from_type());
        } else {
            return false;
        }
    }

    template <typename T>
    auto is() const {
        using raw_t = Traits::extract_raw_type_t<T>;
        return normalized_type_hash == typeid(raw_t).hash_code();
    }

    auto operator==(Type const& other) const noexcept -> bool {
        return normalized_type_hash == other.normalized_type_hash;
    }

    auto operator<(Type const& other) const noexcept -> bool {
        return normalized_type_hash < other.normalized_type_hash;
    }

    auto operator!=(Type const& other) const noexcept -> bool {
        return !(*this == other);
    }

    auto operator<=(Type const& other) const noexcept -> bool {
        return *this < other || *this == other;
    }

    auto operator>(Type const& other) const noexcept -> bool {
        return !(*this <= other);
    }

    auto operator>=(Type const& other) const noexcept -> bool {
        return !(*this < other);
    }

   private:
    Type() = default;
    std::string_view m_outer_type_name{};  // if templated type, this is the outer type name
    std::string_view m_type_name{};        // otherwise, this is the type name

    std::shared_ptr<Type> m_pointed_to_type{};  // if pointer, this is the
    // pointed to type
    std::shared_ptr<Type> m_inherited_from_type{};  // if reflectable
    // type, this is the parent type
    std::shared_ptr<Type> m_contained_type{};  // if array or container, this is
    // the contained type

    std::shared_ptr<Type> m_func_return_type{};
    std::vector<std::shared_ptr<Type>> m_func_arg_types{};
    std::vector<std::shared_ptr<Type>> m_template_args{};
};

/** @brief non-templated field info, not type-safe \n
 * used to dynamically get info about a field or store a type-erased field
 * pointer
 * @note this is a workaround for the fact that we can't have a virtual template
 * function
 */
struct FieldInfo {
    friend struct std::hash<FieldInfo>;

    std::string_view var_name;
    std::string_view type_name;
    Type type;

    // indicates how many fields come before this one in the class
    // does not account for fields in parent classes
    int index;

    auto operator==(FieldInfo const& other) const noexcept -> bool {
        return var_name == other.var_name && type_name == other.type_name && index == other.index;
    }

    auto operator!=(FieldInfo const& other) const noexcept -> bool {
        return !(*this == other);
    }

    auto operator<(FieldInfo const& other) const noexcept -> bool {
        return std::tuple{var_name, type_name, index} <
               std::tuple{other.var_name, other.type_name, other.index};
    }

    auto operator<=(FieldInfo const& other) const noexcept -> bool {
        return *this < other || *this == other;
    }

    auto operator>(FieldInfo const& other) const noexcept -> bool {
        return !(*this <= other);
    }

    auto operator>=(FieldInfo const& other) const noexcept -> bool {
        return !(*this < other);
    }

    template <typename FieldType, typename Reflected>
    auto get(Reflected& obj) -> FieldType& {
        return *static_cast<FieldType*>(m_get(static_cast<void*>(&obj)));
    }

    auto begin() const -> TypeErasedIterator {
        return m_begin;
    }
    auto end() const -> TypeErasedIterator {
        return m_end;
    }

    template <typename TemplatedFieldInfo, typename Reflected,
              typename = std::enable_if_t<Traits::has_tfield_interface_v<TemplatedFieldInfo>>>
    FieldInfo(TemplatedFieldInfo const& info, Reflected& obj)
        : var_name{info.var_name},
          type_name{info.type_name},
          type{Type::of<typename TemplatedFieldInfo::type>()},
          index{info.member_index} {
        using FieldType = typename TemplatedFieldInfo::type;

        m_get = [&info](void* any_obj) {
            auto& ret = const_cast<FieldType&>(info.get(*static_cast<Reflected*>(any_obj)));
            return static_cast<void*>(&ret);
        };

        if constexpr (Traits::is_container<FieldType>::value) {
            auto& container = const_cast<FieldType&>(info.get(obj));
            m_begin = TypeErasedIterator{container, 0};
            m_end = TypeErasedIterator{container, container.size()};
        } else {
            auto& member = const_cast<FieldType&>(info.get(obj));
            m_begin = TypeErasedIterator{member, 0};
            m_end = TypeErasedIterator{member, 1};
        }
    }

   private:
    std::function<void*(void*)> m_get;
    TypeErasedIterator m_begin, m_end;
};

#pragma endregion DynamicTypeInfo

#pragma region Macros

#define STR(x) #x
#define BEGIN_REFLECT_IMPL(_cls, _counter, _parent)                           \
    using _my_type = _cls;                                                    \
    using _parent_type = _parent;                                             \
    static constexpr std::string_view _type_name = STR(_cls);                 \
    static constexpr int _init_count = _counter;                              \
    template <int N, typename F>                                              \
    struct _for_each_field_impl;                                              \
    template <typename F>                                                     \
    struct _for_each_field_impl<-1, F> {                                      \
        static constexpr auto do_it(F&& func) {                               \
            if constexpr (PTS::Traits::is_reflectable_v<_parent_type>) {      \
                PTS::Traits::get_for_each_field_impl<_parent_type, F>::do_it( \
                    std::forward<F>(func));                                   \
            }                                                                 \
        }                                                                     \
    };                                                                        \
    template <int N, auto>                                                    \
    struct _get_field_info_impl

#define FIELD_IMPL(_counter, _var_type, _var_name, _init, ...)                                     \
    _var_type _var_name = _init;                                                                   \
    static constexpr _var_type _my_type::* _##_var_name##_proxy() {                                \
        return &_my_type::_var_name;                                                               \
    }                                                                                              \
    static constexpr auto _##_var_name##_name = #_var_name;                                        \
    static constexpr auto _##_var_name##_modifiers = PTS::ModifierPack{__VA_ARGS__};               \
    static inline auto _##_var_name##_runtime_field_info =                                         \
        PTS::TFieldRuntime<_my_type, _var_type, decltype(_my_type::_##_var_name##_modifiers)>{     \
            _counter - _init_count - 1, _init, _##_var_name##_proxy()};                            \
    static constexpr auto _##_var_name##_field_info =                                              \
        PTS::TField<_my_type, _var_type, decltype(_my_type::_##_var_name##_modifiers)>{            \
            _counter - _init_count - 1, #_var_name, #_var_type, _##_var_name##_runtime_field_info, \
            _##_var_name##_modifiers};                                                             \
    friend struct PTS::TFieldRuntime<_my_type, _var_type,                                          \
                                     decltype(_my_type::_##_var_name##_modifiers)>;                \
    friend struct PTS::TField<_my_type, _var_type, decltype(_my_type::_##_var_name##_modifiers)>;  \
    template <typename F>                                                                          \
    struct _for_each_field_impl<_counter - _init_count - 1, F> {                                   \
        static constexpr auto do_it(F&& func) {                                                    \
            _for_each_field_impl<_counter - _init_count - 2, F>::do_it(std::forward<F>(func));     \
            func(_##_var_name##_field_info);                                                       \
        }                                                                                          \
    };                                                                                             \
    template <int N>                                                                               \
    struct _get_field_info_impl<_counter - _init_count - 1, N> {                                   \
        static constexpr auto do_it() {                                                            \
            return _##_var_name##_field_info;                                                      \
        }                                                                                          \
    };                                                                                             \
    template <int N>                                                                               \
    struct _get_field_info_impl<N, &_my_type::_var_name> {                                         \
        static constexpr auto do_it() {                                                            \
            return _##_var_name##_field_info;                                                      \
        }                                                                                          \
    }

#define END_REFLECT_IMPL(_counter)                                                            \
    static constexpr auto _field_count = _counter - _init_count - 1;                          \
                                                                                              \
   public:                                                                                    \
    static constexpr auto field_count() {                                                     \
        return _field_count;                                                                  \
    }                                                                                         \
    static constexpr auto total_field_count() {                                               \
        return field_count() + PTS::Traits::get_num_members<_parent_type>::value;             \
    }                                                                                         \
    template <typename F>                                                                     \
    static constexpr auto for_each_field(F&& func) {                                          \
        _for_each_field_impl<_field_count - 1, F>::do_it(std::forward<F>(func));              \
    }                                                                                         \
    template <int N>                                                                          \
    static constexpr auto get_field_info() {                                                  \
        return _get_field_info_impl<N, N>::do_it();                                           \
    }                                                                                         \
    template <int N>                                                                          \
    struct callback_data {                                                                    \
        using type = typename decltype(get_field_info<N>())::callback_data_type;              \
    };                                                                                        \
    template <int N>                                                                          \
    using callback_data_t = typename callback_data<N>::type;                                  \
    template <auto MemPtr,                                                                    \
              typename = std::enable_if_t<std::is_member_object_pointer_v<decltype(MemPtr)>>> \
    static constexpr auto get_field_info() {                                                  \
        return _get_field_info_impl<-1, MemPtr>::do_it();                                     \
    }                                                                                         \
    static constexpr auto is_reflectable() {                                                  \
        return true;                                                                          \
    }                                                                                         \
    static constexpr std::string_view const& get_class_name() {                               \
        return _type_name;                                                                    \
    }                                                                                         \
    using parent_type = _parent_type

#define DECL_DYNAMIC_INFO()                                                                      \
   public:                                                                                       \
    static constexpr ClassInfo _class_info = {_type_name, _field_count,                          \
                                              PTS::Traits::get_class_info<_parent_type>::value}; \
    virtual ClassInfo const& dyn_get_class_info() const {                                        \
        return _class_info;                                                                      \
    }                                                                                            \
    virtual std::vector<FieldInfo> dyn_get_field_infos() const {                                 \
        std::vector<FieldInfo> ret;                                                              \
        ret.reserve(_field_count);                                                               \
        _my_type::for_each_field([&ret, this](auto field) { ret.emplace_back(field, *this); });  \
        return ret;                                                                              \
    }                                                                                            \
    template <typename T>                                                                        \
    std::enable_if_t<PTS::Traits::has_dynamic_info<T>::value && std::is_base_of_v<_my_type, T>,  \
                     T*>                                                                         \
    as() {                                                                                       \
        if (dyn_get_class_info().class_name == T::get_class_name()) {                            \
            return dynamic_cast<T*>(this);                                                       \
        }                                                                                        \
        return nullptr;                                                                          \
    }                                                                                            \
    template <typename T>                                                                        \
    std::enable_if_t<PTS::Traits::has_dynamic_info<T>::value && std::is_base_of_v<_my_type, T>,  \
                     T const*>                                                                   \
    as() const {                                                                                 \
        if (dyn_get_class_info().class_name == T::get_class_name()) {                            \
            return dynamic_cast<T const*>(this);                                                 \
        }                                                                                        \
        return nullptr;                                                                          \
    }                                                                                            \
    void dyn_for_each_field(std::function<void(FieldInfo const&)> callable) {                    \
        for (auto& info : dyn_get_field_infos()) {                                               \
            callable(info);                                                                      \
        }                                                                                        \
    }                                                                                            \
    using _warning_silencer = void

#define BEGIN_REFLECT(cls, parent) BEGIN_REFLECT_IMPL(cls, __COUNTER__, parent)
#define FIELD(type, var_name, init, ...) FIELD_IMPL(__COUNTER__, type, var_name, init, __VA_ARGS__)
#define END_REFLECT() END_REFLECT_IMPL(__COUNTER__)

#define DECL_FIELD_EVENT_MEMBERS(method_name, cls, tag)                     \
    void method_name(cls::callback_data_t<tag> data);                       \
    PTS::Callback<void(cls::callback_data_t<tag>)> m_##method_name {        \
        [this](cls::callback_data_t<tag> data) { this->method_name(data); } \
    }

#pragma endregion Macros

#pragma region ReflectionUtil

namespace ReflectionUtil {}  // namespace ReflectionUtil
#pragma endregion ReflectionUtil
}  // namespace PTS

// provide hash functions for std::unordered_map
namespace std {
template <>
struct hash<PTS::ClassInfo> {
    auto operator()(PTS::ClassInfo const& info) const noexcept -> size_t {
        return hash<std::string_view>{}(info.class_name);
    }
};

template <>
struct hash<PTS::FieldInfo> {
    auto operator()(PTS::FieldInfo const& info) const noexcept -> size_t {
        return hash<std::string>{}(std::string{info.type_name} + " " + std::string{info.var_name} +
                                   " " + std::to_string(info.index));
    }
};

template <>
struct hash<PTS::TypeErasedIterator> {
    auto operator()(PTS::TypeErasedIterator const& it) const noexcept -> size_t {
        return hash<std::uintptr_t>{}(reinterpret_cast<std::uintptr_t>(*it));
    }
};

template <>
struct hash<PTS::Type> {
    auto operator()(PTS::Type const& type) const noexcept -> size_t {
        return type.normalized_type_hash;
    }
};
};  // namespace std
