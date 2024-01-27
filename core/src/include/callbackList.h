#pragma once

#include <functional>
#include <forward_list>
#include <vector>

namespace PTS {
	template <typename T>
	struct CallbackList;

	template <typename Ret, typename... Args>
	struct CallbackList<Ret(Args...)>;

	template <typename T>
	struct Callback;

	template <typename Ret, typename... Args>
	struct Callback<Ret(Args...)> {
		friend struct CallbackList<Ret(Args...)>;

		Callback(std::function<Ret(Args...)> func) : m_func{std::move(func)} {}

		template <typename T, typename = std::enable_if_t<
			          std::conjunction_v<
				          std::negation<std::is_same<std::function<Ret(Args...)>, std::decay_t<T>>>,
				          std::is_constructible<std::function<Ret(Args...)>, T&&>>
		          >>
		Callback(T&& arg) : m_func{std::forward<T>(arg)} {}

		template <typename T, typename = std::enable_if_t<
			          std::conjunction_v<
				          std::negation<std::is_same<std::function<Ret(Args...)>, std::decay_t<T>>>,
				          std::is_assignable<std::function<Ret(Args...)>, T&&>>
		          >>
		auto operator=(T&& arg) -> Callback& {
			m_func = std::forward<T>(arg);
			return *this;
		}

		Callback(Callback const&) = delete;
		Callback(Callback&&) = default;
		auto operator=(Callback const&) -> Callback& = delete;
		auto operator=(Callback&&) -> Callback& = default;
		~Callback() = default;

		template <typename... Args2>
		auto operator()(Args2&&... args) const -> decltype(auto) {
			if (m_pfunc) {
				return (*m_pfunc)(std::forward<Args2>(args)...);
			} else {
				return m_func(std::forward<Args2>(args)...);
			}
		}

	private:
		// will be set if the callback is added to a list
		// which is the address of the function in the list (stable in memory)
		std::function<Ret(Args...)>* m_pfunc{nullptr};
		std::function<Ret(Args...)> m_func;
	};

	template <typename Ret, typename... Args>
	struct CallbackList<Ret(Args...)> {
		auto operator+=(Callback<Ret(Args...)>& callback) {
			callback.m_pfunc = &m_callbacks.emplace_front(std::move(callback.m_func));
			++m_size;
		}

		auto operator+=(Callback<Ret(Args...)>&& callback) {
			callback.m_pfunc = &m_callbacks.emplace_front(std::move(callback.m_func));
			++m_size;
		}

		template <typename T, typename = std::enable_if_t<
			          std::conjunction_v<
						  std::negation<std::is_same<Callback<Ret(Args...)>, std::decay_t<T>>>,
				          std::is_constructible<Callback<Ret(Args...)>, T&&>
		          >>>
		auto operator+=(T&& arg) {
			m_callbacks.emplace_front(std::forward<T>(arg));
			++m_size;
		}

		auto operator-=(Callback<Ret(Args...)> const& callback) noexcept {
			if (!callback.m_pfunc) {
				return;
			}

			auto removed{false};
			m_callbacks.remove_if([&](auto&& cb) {
				if (&cb == callback.m_pfunc) {
					removed = true;
					return true;
				}
				return false;
			});

			if (removed) {
				--m_size;
			}
		}

		template <typename... Args2>
		auto operator()(Args2&&... args) const {
			if (!m_callbacks.empty()) {
				// call the function in the original insertion order.
				std::vector<std::function<Ret(Args...)> const*> stk;
				stk.reserve(size());
				for (auto const& cb : m_callbacks) {
					stk.emplace_back(&cb);
				}
				for (auto it = stk.rbegin(); it != stk.rend(); ++it) {
					(**it)(std::forward<Args2>(args)...);
				}
			}
		}

		auto clear() {
			m_callbacks.clear();
		}

		auto size() const {
			return m_size;
		}

		auto empty() const {
			return m_callbacks.empty();
		}

	private:
		size_t m_size{0};
		std::forward_list<std::function<Ret(Args...)>> m_callbacks;
	};
}
