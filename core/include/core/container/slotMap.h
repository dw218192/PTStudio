#pragma once

#include <core/diagnostics.h>

#include <boost/container/flat_map.hpp>
#include <boost/core/span.hpp>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace pts::container {

namespace detail {

/// Compile-time switch: dirty tracking is active iff DirtyMask isn't monostate.
template <class M>
inline constexpr bool dirty_tracking_enabled_v = !std::is_same_v<M, std::monostate>;

template <class M>
constexpr void mask_or_assign(M& dst, M src) noexcept {
    if constexpr (dirty_tracking_enabled_v<M>) {
        using U = std::underlying_type_t<M>;
        dst = static_cast<M>(static_cast<U>(dst) | static_cast<U>(src));
    } else {
        (void) dst;
        (void) src;
    }
}

template <class M>
constexpr M mask_and(M a, M b) noexcept {
    if constexpr (dirty_tracking_enabled_v<M>) {
        using U = std::underlying_type_t<M>;
        return static_cast<M>(static_cast<U>(a) & static_cast<U>(b));
    } else {
        return M{};
    }
}

template <class M>
constexpr M mask_andnot(M a, M b) noexcept {
    if constexpr (dirty_tracking_enabled_v<M>) {
        using U = std::underlying_type_t<M>;
        return static_cast<M>(static_cast<U>(a) & ~static_cast<U>(b));
    } else {
        return M{};
    }
}

template <class M>
constexpr bool mask_is_zero(M m) noexcept {
    if constexpr (dirty_tracking_enabled_v<M>) {
        using U = std::underlying_type_t<M>;
        return static_cast<U>(m) == U{0};
    } else {
        return true;
    }
}

}  // namespace detail

/// Dense slot-map with stable indices, fat-pointer handles, and optional
/// per-entry dirty-mask tracking for cache invalidation.
///
/// Backing storage is a flat vector of Entry structs. Erase tombstones the
/// slot and pushes it onto a free-list for reuse -- indices are never
/// shifted, so handles (and raw indices stored in cross-references like
/// ObjectData::mesh_index) survive unrelated erases and vector reallocation.
///
/// `DirtyMask` is a caller-owned bitmask type (typically a scoped enum with
/// bitwise operators). When DirtyMask is std::monostate (the default) all
/// dirty-tracking machinery compiles away and the container behaves as a
/// plain slot-map. When DirtyMask is a real enum, callers may
/// `register_consumer(subscription)` to receive a bitmap of dirty bits per
/// slot. Mutation sites pass a `changed` mask that is OR'd into each
/// consumer's per-slot bitmap (gated by the consumer's subscription).
/// Consumers `drain_dirty_for(id, query, fn)` to react and clear the
/// observed bits, leaving non-queried bits intact for other readers.
///
/// K must be LessComparable. Compare defaults to std::less<K>; pass
/// std::less<> for transparent (heterogeneous) lookup on string-like keys.
template <class K, class V, class DirtyMask = std::monostate, class Compare = std::less<K>>
class SlotMap {
   public:
    using ConsumerId = uint32_t;

    struct Entry {
        V value{};
        uint64_t version = 0;
        bool active = false;
    };

    struct Handle {
        const SlotMap* cache = nullptr;
        uint32_t idx = UINT32_MAX;

        const V& operator*() const {
            PRECONDITION(cache);
            PRECONDITION(idx < cache->m_entries.size());
            PRECONDITION(cache->m_entries[idx].active);
            return cache->m_entries[idx].value;
        }
        const V* operator->() const {
            return &(**this);
        }
        explicit operator bool() const noexcept {
            return cache != nullptr;
        }
        uint32_t index() const noexcept {
            return idx;
        }
    };

    SlotMap() = default;
    SlotMap(const SlotMap&) = delete;
    SlotMap& operator=(const SlotMap&) = delete;
    SlotMap(SlotMap&&) = default;
    SlotMap& operator=(SlotMap&&) = default;

    /// Insert a new entry. Asserts key is not already present. Marks the
    /// slot dirty for every registered consumer with the consumer's full
    /// subscription bits (i.e. structurally novel).
    Handle insert(K key, V value) {
        PRECONDITION_MSG(!contains(key), "SlotMap::insert: duplicate key");
        uint32_t idx;
        if (!m_free.empty()) {
            idx = m_free.back();
            m_free.pop_back();
            m_entries[idx].value = std::move(value);
            m_entries[idx].version = ++m_next_version;
            m_entries[idx].active = true;
        } else {
            idx = static_cast<uint32_t>(m_entries.size());
            m_entries.push_back(Entry{std::move(value), ++m_next_version, true});
            grow_consumers(m_entries.size());
        }
        on_lifecycle_change(idx);
        m_index.emplace(std::move(key), idx);
        return Handle{this, idx};
    }

    /// Replace value at handle, bump version (globally monotonic). Marks
    /// the slot dirty with the supplied mask for each consumer.
    void upsert(Handle h, V new_value, DirtyMask changed = DirtyMask{}) {
        PRECONDITION(h.cache == this);
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        m_entries[h.idx].value = std::move(new_value);
        m_entries[h.idx].version = ++m_next_version;
        propagate_dirty(h.idx, changed);
    }

    /// In-place mutation; bumps version (globally monotonic) after fn returns.
    /// `changed` indicates which fields the callback touched -- OR'd into each
    /// consumer's per-slot dirty bits (gated by the consumer's subscription).
    template <class Fn>
    void mutate(Handle h, DirtyMask changed, Fn&& fn) {
        PRECONDITION(h.cache == this);
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        std::forward<Fn>(fn)(m_entries[h.idx].value);
        m_entries[h.idx].version = ++m_next_version;
        propagate_dirty(h.idx, changed);
    }

    /// Convenience overload for monostate (no dirty tracking). Removed from
    /// the overload set when dirty tracking is enabled, forcing callers to
    /// declare which fields they touched.
    template <class Fn, class M = DirtyMask,
              std::enable_if_t<std::is_same_v<M, std::monostate>, int> = 0>
    void mutate(Handle h, Fn&& fn) {
        mutate(h, DirtyMask{}, std::forward<Fn>(fn));
    }

    /// Find entry by key. Returns invalid Handle if not present.
    template <class K2>
    Handle find(const K2& key) const {
        auto it = m_index.find(key);
        if (it == m_index.end()) return {};
        return Handle{this, it->second};
    }

    template <class K2>
    bool contains(const K2& key) const {
        return m_index.find(key) != m_index.end();
    }

    /// Tombstone entry and push to free-list. Resets value to release
    /// RAII resources (GPU handles etc.) immediately. Marks the slot
    /// dirty for every consumer (their full subscription) so consumers
    /// observe the disappearance during the next drain.
    template <class K2>
    void erase(const K2& key) {
        auto it = m_index.find(key);
        if (it == m_index.end()) return;
        auto idx = it->second;
        m_entries[idx].value = V{};
        m_entries[idx].active = false;
        m_free.push_back(idx);
        m_index.erase(it);
        on_lifecycle_change(idx);
    }

    uint64_t version(Handle h) const {
        PRECONDITION(h.cache == this);
        PRECONDITION(h.idx < m_entries.size());
        return m_entries[h.idx].version;
    }

    // -- Index-based access (for cross-references and GPU slots) --

    const V& at(uint32_t idx) const {
        PRECONDITION(idx < m_entries.size());
        PRECONDITION(m_entries[idx].active);
        return m_entries[idx].value;
    }

    bool active_at(uint32_t idx) const {
        if (idx >= m_entries.size()) return false;
        return m_entries[idx].active;
    }

    uint64_t version_at(uint32_t idx) const {
        PRECONDITION(idx < m_entries.size());
        return m_entries[idx].version;
    }

    /// In-place mutation by raw index; bumps version (globally monotonic).
    template <class Fn>
    void mutate_at(uint32_t idx, DirtyMask changed, Fn&& fn) {
        PRECONDITION(idx < m_entries.size());
        PRECONDITION(m_entries[idx].active);
        std::forward<Fn>(fn)(m_entries[idx].value);
        m_entries[idx].version = ++m_next_version;
        propagate_dirty(idx, changed);
    }

    /// Convenience overload for monostate users.
    template <class Fn, class M = DirtyMask,
              std::enable_if_t<std::is_same_v<M, std::monostate>, int> = 0>
    void mutate_at(uint32_t idx, Fn&& fn) {
        mutate_at(idx, DirtyMask{}, std::forward<Fn>(fn));
    }

    // -- Iteration --

    /// Iterate active entries. Callback: fn(const K& key, V& value).
    template <class Fn>
    void for_each(Fn&& fn) {
        for (auto& [key, idx] : m_index) {
            fn(key, m_entries[idx].value);
        }
    }

    /// Iterate active entries (const). Callback: fn(const K& key, const V& value).
    template <class Fn>
    void for_each(Fn&& fn) const {
        for (const auto& [key, idx] : m_index) {
            fn(key, m_entries[idx].value);
        }
    }

    /// Raw backing vector including tombstoned holes. Use for index-based
    /// GPU iteration where the slot index must match the buffer position.
    boost::span<const Entry> span_raw() const {
        return {m_entries.data(), m_entries.size()};
    }

    /// Number of live (active) entries.
    size_t size() const noexcept {
        return m_index.size();
    }

    /// Total vector capacity (live + tombstoned).
    size_t capacity() const noexcept {
        return m_entries.size();
    }

    void clear() {
        m_entries.clear();
        m_index.clear();
        m_free.clear();
        for (auto& c : m_consumers) {
            c.per_slot_dirty.clear();
            c.aggregate = DirtyMask{};
        }
        // m_next_version intentionally NOT reset -- monotonic across clears
    }

    // -- Dirty-mask tracking (no-op when DirtyMask is std::monostate) --

    /// Register a new consumer interested in the given subscription bits.
    /// The returned id is stable for the lifetime of the SlotMap.
    /// Newly registered consumers start fully dirty: the aggregate is set
    /// to `subscription` (even when the map is empty), and every active
    /// slot's per-slot bitmap is primed with `subscription`. The first
    /// drain therefore always triggers a rebuild path -- matching the
    /// classic "version starts mismatched" idiom.
    ConsumerId register_consumer(DirtyMask subscription) {
        ConsumerState state;
        state.subscription = subscription;
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            state.per_slot_dirty.assign(m_entries.size(), DirtyMask{});
            for (size_t i = 0; i < m_entries.size(); ++i) {
                if (m_entries[i].active) {
                    state.per_slot_dirty[i] = subscription;
                }
            }
            state.aggregate = subscription;
        }
        m_consumers.push_back(std::move(state));
        return static_cast<ConsumerId>(m_consumers.size() - 1);
    }

    /// True if any slot has at least one queried bit set for this consumer.
    /// O(1) -- consults the cached aggregate.
    bool any_dirty_for(ConsumerId id, DirtyMask query) const {
        PRECONDITION(id < m_consumers.size());
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            return !detail::mask_is_zero(detail::mask_and(m_consumers[id].aggregate, query));
        } else {
            (void) id;
            (void) query;
            return false;
        }
    }

    /// Walk every slot whose per-slot dirty bits intersect `query`,
    /// invoking `fn(uint32_t slot_idx, const V& value)`. Slots may be
    /// inactive (recently erased) -- callbacks should consult `active_at`
    /// when that distinction matters. The intersected bits are cleared
    /// after the callback; bits outside `query` are left intact for other
    /// drain calls or other queries.
    template <class Fn>
    void drain_dirty_for(ConsumerId id, DirtyMask query, Fn&& fn) {
        PRECONDITION(id < m_consumers.size());
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            auto& c = m_consumers[id];
            DirtyMask remaining = DirtyMask{};
            for (uint32_t i = 0; i < c.per_slot_dirty.size(); ++i) {
                DirtyMask slot_bits = c.per_slot_dirty[i];
                if (!detail::mask_is_zero(detail::mask_and(slot_bits, query))) {
                    fn(i, m_entries[i].value);
                    slot_bits = detail::mask_andnot(slot_bits, query);
                    c.per_slot_dirty[i] = slot_bits;
                }
                detail::mask_or_assign(remaining, slot_bits);
            }
            c.aggregate = remaining;
        } else {
            (void) id;
            (void) query;
            (void) fn;
        }
    }

    /// Read the consumer's subscription mask. Useful for assertions and
    /// to recover what bits the consumer registered for.
    DirtyMask subscription_for(ConsumerId id) const {
        PRECONDITION(id < m_consumers.size());
        return m_consumers[id].subscription;
    }

   private:
    struct ConsumerState {
        DirtyMask subscription{};
        std::vector<DirtyMask> per_slot_dirty;
        DirtyMask aggregate{};
    };

    void grow_consumers(size_t new_capacity) {
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            for (auto& c : m_consumers) {
                if (c.per_slot_dirty.size() < new_capacity) {
                    c.per_slot_dirty.resize(new_capacity, DirtyMask{});
                }
            }
        } else {
            (void) new_capacity;
        }
    }

    /// Mark a slot dirty for all consumers using the consumer's full
    /// subscription mask. Used on insert/erase where the slot's lifecycle
    /// changed -- consumers should re-scan regardless of which fields care.
    void on_lifecycle_change(uint32_t idx) {
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            for (auto& c : m_consumers) {
                if (c.per_slot_dirty.size() <= idx) {
                    c.per_slot_dirty.resize(idx + 1, DirtyMask{});
                }
                detail::mask_or_assign(c.per_slot_dirty[idx], c.subscription);
                detail::mask_or_assign(c.aggregate, c.subscription);
            }
        } else {
            (void) idx;
        }
    }

    void propagate_dirty(uint32_t idx, DirtyMask changed) {
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            for (auto& c : m_consumers) {
                DirtyMask relevant = detail::mask_and(changed, c.subscription);
                if (detail::mask_is_zero(relevant)) continue;
                if (c.per_slot_dirty.size() <= idx) {
                    c.per_slot_dirty.resize(idx + 1, DirtyMask{});
                }
                detail::mask_or_assign(c.per_slot_dirty[idx], relevant);
                detail::mask_or_assign(c.aggregate, relevant);
            }
        } else {
            (void) idx;
            (void) changed;
        }
    }

    std::vector<Entry> m_entries;
    boost::container::flat_map<K, uint32_t, Compare> m_index;
    std::vector<uint32_t> m_free;
    std::vector<ConsumerState> m_consumers;
    uint64_t m_next_version = 0;
};

}  // namespace pts::container
