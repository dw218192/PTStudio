#pragma once
#include "utils.h"
#include "callbackList.h"
#include <array>
#include <unordered_map>
#include <string>

namespace PTS {
	template <typename GPUType, typename CPUType, typename UserDataType, size_t N>
	struct ContinuousGPUBufferLink {
		using cpu_handle_t = CPUType const*;
		using on_erase_callback_fn_t = void(UserDataType& user_data, cpu_handle_t handle, GPUType const& erased_val);
		using on_push_back_callback_fn_t = void(UserDataType& user_data, cpu_handle_t handle, GPUType const& new_val);
		using on_update_callback_fn_t = void(UserDataType& user_data, cpu_handle_t handle, GPUType const& new_val);

		ContinuousGPUBufferLink() = default;
		~ContinuousGPUBufferLink() = default;
		ContinuousGPUBufferLink(UserDataType user_data) noexcept : m_user_data{std::move(user_data)} {}
		DEFAULT_COPY_MOVE(ContinuousGPUBufferLink);

		struct Proxy {
			cpu_handle_t cpu_handle;
			std::reference_wrapper<ContinuousGPUBufferLink> link;
			std::reference_wrapper<GPUType> gpu_handle;

			Proxy(ContinuousGPUBufferLink& link, cpu_handle_t cpu_handle, GPUType& gpu_handle) noexcept :
				link{link}, cpu_handle{cpu_handle}, gpu_handle{gpu_handle} {}

			auto operator=(GPUType& value) noexcept -> Proxy& {
				gpu_handle.get() = value;
				link.get().m_on_update_callbacks(link.get().m_user_data, cpu_handle, gpu_handle.get());
				return *this;
			}

			auto operator=(GPUType&& value) noexcept -> Proxy& {
				gpu_handle.get() = std::move(value);
				link.get().m_on_update_callbacks(link.get().m_user_data, cpu_handle, gpu_handle.get());
				return *this;
			}

			operator GPUType&() noexcept { return gpu_handle; }
			operator GPUType const&() const noexcept { return gpu_handle; }
		};

		template <typename T, typename = std::enable_if_t<
			          std::disjunction_v<
				          std::is_same<std::decay_t<T>, GPUType>,
				          std::is_assignable<GPUType&, T>
			          >
		          >>
		NODISCARD auto push_back(cpu_handle_t cpu_handle, T&& value) noexcept -> tl::expected<size_t, std::string>;
		NODISCARD auto erase(cpu_handle_t cpu_handle) noexcept -> tl::expected<void, std::string>;
		NODISCARD auto get_idx(cpu_handle_t cpu_handle) const noexcept -> tl::expected<size_t, std::string>;
		NODISCARD auto get_ins(size_t idx) const noexcept -> tl::expected<cpu_handle_t, std::string>;

		NODISCARD auto data() noexcept { return m_data.data(); }
		NODISCARD auto data() const noexcept { return m_data.data(); }
		NODISCARD auto size() const noexcept -> size_t { return m_size; }
		NODISCARD auto empty() const noexcept -> bool { return !m_size; }
		auto clear() noexcept -> void;

		auto begin() noexcept { return m_data.begin(); }
		auto begin() const noexcept { return m_data.begin(); }
		auto end() noexcept { return m_data.begin() + m_size; }
		auto end() const noexcept { return m_data.begin() + m_size; }

		NODISCARD auto operator[](size_t idx) -> Proxy {
			return Proxy{*this, m_idx_to_ins.at(idx), m_data.at(idx)};
		}

		NODISCARD auto at(size_t idx) const -> GPUType const& { return m_data.at(idx); }
		NODISCARD auto get_on_erase_callbacks() noexcept -> auto& { return m_on_erase_callbacks; }
		NODISCARD auto get_on_push_back_callbacks() noexcept -> auto& { return m_on_push_back_callbacks; }
		NODISCARD auto get_on_update_callbacks() noexcept -> auto& { return m_on_update_callbacks; }
		NODISCARD auto get_user_data() noexcept -> auto& { return m_user_data; }

	private:
		size_t m_size{0};
		std::array<GPUType, N> m_data;
		std::unordered_map<cpu_handle_t, size_t> m_ins_to_idx;
		std::unordered_map<size_t, cpu_handle_t> m_idx_to_ins;
		UserDataType m_user_data;

		CallbackList<on_erase_callback_fn_t> m_on_erase_callbacks;
		CallbackList<on_push_back_callback_fn_t> m_on_push_back_callbacks;
		CallbackList<on_update_callback_fn_t> m_on_update_callbacks;
	};

	template <typename GPUType, typename CPUType, typename UserDataType, size_t N>
	template <typename T, typename>
	auto ContinuousGPUBufferLink<GPUType, CPUType, UserDataType, N>::push_back(
		cpu_handle_t cpu_handle, T&& value) noexcept -> tl::expected<size_t, std::string> {
		if (m_size >= N) {
			return TL_ERROR("Buffer is full");
		}
		if (m_ins_to_idx.count(cpu_handle)) {
			return TL_ERROR("CPU handle already exists");
		}

		auto const idx = m_size++;
		m_data[idx] = std::forward<T>(value);
		m_ins_to_idx[cpu_handle] = idx;
		m_idx_to_ins[idx] = cpu_handle;
		m_on_push_back_callbacks(get_user_data(), cpu_handle, m_data[idx]);
		return {};
	}

	template <typename GPUType, typename CPUType, typename UserDataType, size_t N>
	auto ContinuousGPUBufferLink<GPUType, CPUType, UserDataType, N>::erase(
		cpu_handle_t cpu_handle) noexcept -> tl::expected<void, std::string> {
		auto const it = m_ins_to_idx.find(cpu_handle);
		if (it == m_ins_to_idx.end()) {
			return TL_ERROR("CPU handle not found");
		}

		// swap with last element
		auto const idx = it->second;
		auto const last_idx = m_size - 1;
		try {
			m_data[idx] = std::move(m_data[last_idx]);
			auto last_ins = m_idx_to_ins.at(last_idx);
			m_ins_to_idx.at(last_ins) = idx;
			m_idx_to_ins.at(idx) = last_ins;

			m_ins_to_idx.erase(it);
			m_idx_to_ins.erase(last_idx);
		} catch (std::out_of_range const& err) {
			return TL_ERROR("Out of range, attempting to swap {} with {}: {}",
			                idx, m_size - 1, err.what());
		}

		--m_size;
		m_on_erase_callbacks(get_user_data(), cpu_handle, m_data[idx]);
		return {};
	}

	template <typename GPUType, typename CPUType, typename UserDataType, size_t N>
	auto ContinuousGPUBufferLink<GPUType, CPUType, UserDataType, N>::get_idx(
		cpu_handle_t cpu_handle) const noexcept -> tl::expected<size_t, std::string> {
		auto const it = m_ins_to_idx.find(cpu_handle);
		if (it == m_ins_to_idx.end()) {
			return TL_ERROR("CPU handle not found");
		}
		if (it->second >= m_size) {
			return TL_ERROR("Index out of range");
		}

		return it->second;
	}

	template <typename GPUType, typename CPUType, typename UserDataType, size_t N>
	auto ContinuousGPUBufferLink<GPUType, CPUType, UserDataType, N>::get_ins(
		size_t idx) const noexcept -> tl::expected<cpu_handle_t, std::string> {
		auto const it = m_idx_to_ins.find(idx);
		if (it == m_idx_to_ins.end()) {
			return TL_ERROR("Index not found");
		}
		if (!it->second) {
			return TL_ERROR("CPU handle is null");
		}

		return it->second;
	}

	template <typename GPUType, typename CPUType, typename UserDataType, size_t N>
	auto ContinuousGPUBufferLink<GPUType, CPUType, UserDataType, N>::clear() noexcept -> void {
		m_size = 0;
		m_ins_to_idx.clear();
		m_idx_to_ins.clear();
	}
} // namespace PTS
