#pragma once
#include <cstddef>
#include <new>
#include <utility>

namespace pts {

/// CRTP base providing in-place PIMPL storage -- embedded aligned byte buffer,
/// no heap allocation. Derived classes:
///   1. Declare Impl as a namespace-scope forward declaration (NOT nested),
///      because Impl is a base-class template argument.
///   2. Declare all five special members in the header and define them in the
///      .cpp where Impl is complete. Derived ctor calls `construct(...)`;
///      derived dtor MUST call `destroy()`. Move ctor/assign must forward to
///      construct/impl as appropriate.
///   3. Must NOT hold pointers to Impl-internal members or to the outer class
///      from inside Impl -- the outer object (and thus the embedded buffer)
///      moves with move operations, so self-pointers become dangling.
///      If address stability is required, use heap PIMPL instead.
///   4. Should be marked `final`. Private inheritance from InPlacePimpl is an
///      implementation detail, not a polymorphism point.
///   5. Size should be sized for worst-case STL (MSVC Debug containers are
///      larger than Release); CI builds both configs so the static_assert
///      fires in either. Round up with headroom (typical: next power of two).
template <typename Derived, typename Impl, std::size_t Size, std::size_t Align>
class InPlacePimpl {
   protected:
    template <typename... Args>
    void construct(Args&&... args) {
        static_assert(sizeof(Impl) <= Size, "InPlacePimpl: Size too small -- grow Size");
        static_assert(alignof(Impl) <= Align, "InPlacePimpl: Align too small -- grow Align");
        ::new (static_cast<void*>(m_storage)) Impl(std::forward<Args>(args)...);
    }

    void destroy() noexcept {
        impl().~Impl();
    }

    Impl& impl() noexcept {
        return *std::launder(reinterpret_cast<Impl*>(m_storage));
    }
    const Impl& impl() const noexcept {
        return *std::launder(reinterpret_cast<const Impl*>(m_storage));
    }

   private:
    alignas(Align) std::byte m_storage[Size];
};

}  // namespace pts
