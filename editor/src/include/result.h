#pragma once
#include <variant>
#include <optional>
#include <string>
#include <utility>

template <typename T, typename E = std::string>
struct Result {
    constexpr Result(T&& t) noexcept : m_data{ std::move(t) } {}
    constexpr Result(E&& e) noexcept : m_data{ std::move(e) } {}
    constexpr Result(T const& t) noexcept : m_data{ t } {}
    constexpr Result(E const& e) noexcept : m_data{ e } {}

    [[nodiscard]] constexpr auto valid() const noexcept { return std::holds_alternative<T>(m_data); }

	[[nodiscard]] constexpr auto value() const -> T const& { return std::get<0>(m_data); }
    [[nodiscard]] constexpr auto value() -> T& { return std::get<0>(m_data); }

    [[nodiscard]] constexpr auto error() const -> E const& { return std::get<1>(m_data); }
    
private:
    std::variant<T, E> m_data;
};

template <typename E>
struct Result<void, E> {
    static inline constexpr Result<void, E> ok() noexcept { return Result<void, E>{}; }
    explicit constexpr Result() noexcept : m_data{ std::nullopt } {}
    constexpr Result(E&& e) noexcept : m_data{ std::move(e) } {}
    constexpr Result(E const& e) noexcept : m_data{ e } {}

    [[nodiscard]] constexpr auto valid() const noexcept -> bool { return !m_data; }
    [[nodiscard]] constexpr auto error() const -> E const& { return *m_data; }
    static void value() noexcept { }
private:
    std::optional<E> m_data;
};

template <typename T, typename E>
struct Result<T&, E> {
    using RefType = std::reference_wrapper<T>;

    constexpr Result(T& t) noexcept : m_data{ std::ref(t) } {}
    constexpr Result(E&& e) noexcept : m_data{ std::move(e) } {}
    constexpr Result(E const& e) noexcept : m_data{ e } {}

    [[nodiscard]] constexpr auto valid() const noexcept { return std::holds_alternative<RefType>(m_data); }
    [[nodiscard]] constexpr auto value() -> T& { return std::get<0>(m_data); }
    [[nodiscard]] constexpr auto error() const -> E const& { return std::get<1>(m_data); }
    std::variant<RefType, E> m_data;
};
template <typename T, typename E>
struct Result<T const&, E> {
    using RefType = std::reference_wrapper<T const>;
    constexpr Result(T const& t) noexcept : m_data{ std::cref(t) } {}
    constexpr Result(E&& e) noexcept : m_data{ std::move(e) } {}
    constexpr Result(E const& e) noexcept : m_data{ e } {}

    [[nodiscard]] constexpr auto valid() const noexcept { return std::holds_alternative<RefType>(m_data); }
    [[nodiscard]] constexpr auto value() const -> T const& { return std::get<0>(m_data); }
    [[nodiscard]] constexpr auto error() const -> E const& { return std::get<1>(m_data); }
    std::variant<RefType, E> m_data;
};
template <typename T>
Result<T> GLErrorResult(unsigned code) {
    return std::string{ "OpenGL error code: " } + std::to_string(code);
}