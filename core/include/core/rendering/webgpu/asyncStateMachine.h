#pragma once

#include <core/diagnostics.h>
#include <webgpu/webgpu.h>

#include <thread>
#include <variant>

namespace pts::webgpu {

/// CRTP base providing reusable async WebGPU state machine infrastructure.
///
/// Derived must provide:
///   - void on_tick()                         — called after event processing each tick
///   - bool is_pending() const                — true while async callbacks are in flight
///   - WGPUInstance wgpu_instance() const      — instance handle for event processing
template <typename Derived, typename... States>
class AsyncStateMachine {
   protected:
    std::variant<States...> m_state;

    // -- state queries --------------------------------------------------------

    template <typename S>
    bool is() const {
        return std::holds_alternative<S>(m_state);
    }

    template <typename S>
    S& get() {
        return std::get<S>(m_state);
    }

    template <typename S>
    const S& get() const {
        return std::get<S>(m_state);
    }

    template <typename S>
    S* get_if() {
        return std::get_if<S>(&m_state);
    }

    template <typename S>
    const S* get_if() const {
        return std::get_if<S>(&m_state);
    }

    // -- transitions ----------------------------------------------------------

    template <typename S, typename... Args>
    void transition(Args&&... args) {
        m_state.template emplace<S>(std::forward<Args>(args)...);
    }

    // -- tick / poll ----------------------------------------------------------

    void tick() {
        auto* self = static_cast<Derived*>(this);
        WGPUInstance inst = self->wgpu_instance();
        if (inst) {
            wgpuInstanceProcessEvents(inst);
        }
        self->on_tick();
    }

    void tick_until_settled() {
#ifdef __EMSCRIPTEN__
        INVARIANT_MSG(false, "synchronous blocking is not allowed on Emscripten");
#else
        auto* self = static_cast<Derived*>(this);
        while (self->is_pending()) {
            tick();
            std::this_thread::yield();
        }
#endif
    }

    // -- lifecycle ------------------------------------------------------------

    AsyncStateMachine() = default;
    ~AsyncStateMachine() = default;

    AsyncStateMachine(const AsyncStateMachine&) = delete;
    AsyncStateMachine& operator=(const AsyncStateMachine&) = delete;

    AsyncStateMachine(AsyncStateMachine&& other) noexcept {
        INVARIANT(!static_cast<const Derived*>(&other)->is_pending());
        m_state = std::move(other.m_state);
    }

    AsyncStateMachine& operator=(AsyncStateMachine&& other) noexcept {
        INVARIANT(!static_cast<const Derived*>(&other)->is_pending());
        m_state = std::move(other.m_state);
        return *this;
    }
};

}  // namespace pts::webgpu
