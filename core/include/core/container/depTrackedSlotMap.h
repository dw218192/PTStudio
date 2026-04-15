#pragma once

#include <core/container/slotMap.h>
#include <core/diagnostics.h>

#include <boost/container/small_vector.hpp>
#include <boost/core/span.hpp>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

namespace pts::container {

/// Dependency-tracked cache built on SlotMap.
///
/// Each entry carries a snapshot of the dep-version vector it was built
/// under.  `get_or_build` rebuilds when any dep changed, when
/// `invalidate(key)` was called, or on first access (cache miss).
///
/// Compare defaults to std::less<K>; pass std::less<> for transparent
/// (heterogeneous) lookup on string-like keys.
template <class K, class V, class Compare = std::less<K>>
class DepTrackedSlotMap {
   public:
    using Span = boost::span<const uint64_t>;

    struct Tracked {
        V value{};
        boost::container::small_vector<uint64_t, 4> deps_snapshot;
        bool dirty = false;
    };

    using InnerMap = SlotMap<K, Tracked, Compare>;

    struct Handle {
        typename InnerMap::Handle m_inner;

        const V& operator*() const {
            return (*m_inner).value;
        }
        const V* operator->() const {
            return &(*m_inner).value;
        }
        explicit operator bool() const noexcept {
            return static_cast<bool>(m_inner);
        }
        uint32_t index() const noexcept {
            return m_inner.index();
        }
    };

    DepTrackedSlotMap() = default;
    DepTrackedSlotMap(const DepTrackedSlotMap&) = delete;
    DepTrackedSlotMap& operator=(const DepTrackedSlotMap&) = delete;
    DepTrackedSlotMap(DepTrackedSlotMap&&) = default;
    DepTrackedSlotMap& operator=(DepTrackedSlotMap&&) = default;

    /// Build on miss or when deps/dirty changed. Returns handle that
    /// derefs to const V&.
    template <class K2, class BuildFn>
    Handle get_or_build(const K2& key, Span current_deps, BuildFn&& build) {
        auto h = m_impl.find(key);
        if (!h) {
            Tracked t;
            t.value = std::forward<BuildFn>(build)();
            t.deps_snapshot.assign(current_deps.begin(), current_deps.end());
            auto inner = m_impl.insert(K(key), std::move(t));
            return Handle{inner};
        }
        const Tracked& existing = *h;
        if (!existing.dirty && deps_match(existing.deps_snapshot, current_deps)) {
            return Handle{h};
        }
        m_impl.mutate(h, [&](Tracked& t) {
            t.value = std::forward<BuildFn>(build)();
            t.deps_snapshot.assign(current_deps.begin(), current_deps.end());
            t.dirty = false;
        });
        return Handle{h};
    }

    /// Build on miss or dep/dirty mismatch. Calls on_replace(old_value)
    /// BEFORE writing the new value (for releasing GPU handles).
    template <class K2, class BuildFn, class OnReplaceFn>
    Handle get_or_build_with_replace(const K2& key, Span current_deps, BuildFn&& build,
                                     OnReplaceFn&& on_replace) {
        auto h = m_impl.find(key);
        if (!h) {
            Tracked t;
            t.value = std::forward<BuildFn>(build)();
            t.deps_snapshot.assign(current_deps.begin(), current_deps.end());
            auto inner = m_impl.insert(K(key), std::move(t));
            return Handle{inner};
        }
        const Tracked& existing = *h;
        if (!existing.dirty && deps_match(existing.deps_snapshot, current_deps)) {
            return Handle{h};
        }
        m_impl.mutate(h, [&](Tracked& t) {
            std::forward<OnReplaceFn>(on_replace)(t.value);
            t.value = std::forward<BuildFn>(build)();
            t.deps_snapshot.assign(current_deps.begin(), current_deps.end());
            t.dirty = false;
        });
        return Handle{h};
    }

    /// Version of the entry identified by handle.
    uint64_t version(Handle h) const {
        return m_impl.version(h.m_inner);
    }

    /// Version by key lookup. Returns 0 if not present.
    template <class K2>
    uint64_t version(const K2& key) const {
        auto h = m_impl.find(key);
        if (!h) return 0;
        return m_impl.version(h);
    }

    /// Find by key. Returns invalid Handle if not present.
    template <class K2>
    Handle find(const K2& key) const {
        return Handle{m_impl.find(key)};
    }

    template <class K2>
    bool contains(const K2& key) const {
        return m_impl.contains(key);
    }

    /// Force next get_or_build for this key to rebuild.
    template <class K2>
    void invalidate(const K2& key) {
        auto h = m_impl.find(key);
        if (!h) return;
        m_impl.mutate(h, [](Tracked& t) {
            t.dirty = true;
            // mutate bumps version but that is harmless here -- the rebuild
            // on the next get_or_build will bump it again with the real value.
        });
    }

    void invalidate_all() {
        m_impl.for_each([](const K&, Tracked& t) { t.dirty = true; });
    }

    template <class K2>
    void erase(const K2& key) {
        m_impl.erase(key);
    }

    /// Iterate entries. Callback: fn(const K& key, const V& value).
    template <class Fn>
    void for_each(Fn&& fn) const {
        m_impl.for_each([&](const K& key, const Tracked& t) { fn(key, t.value); });
    }

    /// Mutable iteration. Callback: fn(const K& key, V& value).
    template <class Fn>
    void for_each(Fn&& fn) {
        m_impl.for_each([&](const K& key, Tracked& t) { fn(key, t.value); });
    }

    void clear() {
        m_impl.clear();
    }

    size_t size() const noexcept {
        return m_impl.size();
    }

    bool empty() const noexcept {
        return m_impl.size() == 0;
    }

   private:
    static bool deps_match(const boost::container::small_vector<uint64_t, 4>& snapshot,
                           Span current) {
        if (snapshot.size() != current.size()) return false;
        for (size_t i = 0; i < current.size(); ++i) {
            if (snapshot[i] != current[i]) return false;
        }
        return true;
    }

    InnerMap m_impl;
};

}  // namespace pts::container
