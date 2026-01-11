#pragma once
#include <stdexcept>

namespace PTS {
template <class Derived>
struct Singleton {
    NO_COPY_MOVE(Singleton);

    static auto get() -> Derived& {
        return *p;
    }
    // Create the singleton instance
    template <typename... Args>
    static auto create(Args&&... args) -> Derived& {
        if (p) {
            throw std::runtime_error("Cannot create more than one instance of a singleton type");
        }
        p = new Derived{std::forward<Args>(args)...};
        return *p;
    }

   protected:
    static inline Derived* p{nullptr};
    Singleton() = default;
};
}  // namespace PTS