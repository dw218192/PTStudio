#pragma once
#include "utils.h"

namespace PTS {
    template <typename Derived>
    struct Singleton {
        NO_COPY_MOVE(Singleton);
        
        static auto get() -> Derived& {
            return *m_pins;
        }
        // Create the singleton instance
        template<typename... Args>
        static auto create(Args&&... args) noexcept -> tl::expected<void, std::string> {
            if (m_pins) {
                return TL_ERROR("Cannot create more than one instance of a singleton type");
            }
            m_pins = new Derived(std::forward<Args>(args)...);
            return {};
        }
    protected:
        static inline Derived* m_pins { nullptr };
        Singleton() = default;
    };

    template <typename Derived, typename = std::enable_if_t<std::is_default_constructible_v<Derived>>>
    struct LazySingleton {
        NO_COPY_MOVE(LazySingleton);
        
        static auto get() -> Derived& {
            static Derived ins;
            return ins;
        }
    protected:
        LazySingleton() = default;
    };
}