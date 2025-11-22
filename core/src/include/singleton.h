#pragma once
#include "utils.h"
namespace PTS {
template <class Derived>
struct Singleton {
    NO_COPY_MOVE(Singleton);

    static auto get() -> Derived& {
        return *p;
    }
    // Create the singleton instance
    template <typename... Args>
    static auto create(Args&&... args) noexcept -> tl::expected<void, std::string> {
        if (p) {
            return TL_ERROR("Cannot create more than one instance of a singleton type");
        }
        p = new Derived(std::forward<Args>(args)...);
        return {};
    }

   protected:
    static inline Derived* p{nullptr};
    Singleton() = default;
};
}  // namespace PTS