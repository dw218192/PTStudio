#pragma once

#include <core/diagnostics.h>

#include <boost/core/span.hpp>
#include <cstdint>
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

/// Dense slot allocator with stable indices, free-list reuse, and optional
/// per-field dirty-mask tracking for cache invalidation.
///
/// Storage is a flat vector of Entry structs. Erase tombstones the slot and
/// pushes its index onto a free-list; indices never shift, so cross-references
/// (e.g. ObjectData::mesh_index) survive unrelated erases and reallocation.
///
/// `DirtyMask` is a caller-owned bitmask type (typically a scoped enum with
/// bitwise operators). When DirtyMask is std::monostate (the default) all
/// dirty-tracking machinery compiles away and the container behaves as a plain
/// slot allocator.
///
/// ### Lifecycle semantics
///
/// Insert implicitly stamps the new slot with each consumer's full subscription
/// mask, and sets a per-consumer `inserts_pending` flag surfaced via
/// `any_inserts_for()`. A consumer's next drain observes a brand-new slot as
/// fully dirty and does first-time upload naturally -- no "Lifecycle" bit is
/// required. `any_inserts_for()` lets callers that cache a packed GPU buffer
/// (i.e. not keyed by slot index) detect that a structural rebuild is needed.
///
/// Erase queues the slot index on every consumer's erase-event list and clears
/// any pending per-slot dirty bits for that slot (since the slot is now
/// inactive, no on_update should fire for it). The two-callback `drain()` API
/// surfaces erase events via the `on_erase` callback before any `on_update`
/// callbacks run, so a consumer may release per-slot GPU state before the
/// slot can be observed as active again via free-list reuse.
///
/// `register_consumer` primes `inserts_pending = true` so the first drain
/// always triggers a full-rebuild path (matching the classic "version starts
/// mismatched" idiom). `drain()` clears both the erase-event list and the
/// inserts_pending flag.
template <class V, class DirtyMask = std::monostate>
class SlotArray {
   public:
    using ConsumerId = uint32_t;

    struct Entry {
        V value{};
        uint64_t version = 0;
        bool active = false;
    };

    struct Handle {
        uint32_t idx = UINT32_MAX;

        explicit operator bool() const noexcept {
            return idx != UINT32_MAX;
        }
        uint32_t index() const noexcept {
            return idx;
        }
    };

    SlotArray() = default;
    SlotArray(const SlotArray&) = delete;
    SlotArray& operator=(const SlotArray&) = delete;
    SlotArray(SlotArray&&) = default;
    SlotArray& operator=(SlotArray&&) = default;

    /// Insert a new entry; reuses a free-listed slot if available. Stamps
    /// the new slot with each consumer's full subscription mask, so first-
    /// time observers see the slot as fully dirty on the next drain.
    Handle insert(V value) {
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
        on_insert(idx);
        return Handle{idx};
    }

    /// Tombstone a slot and queue an erase event on each consumer. Resets
    /// the value to release RAII-owned resources immediately.
    void erase(Handle h) {
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        erase_at(h.idx);
    }

    /// Tombstone by raw slot index. Convenience for call sites that hold an
    /// index rather than a Handle.
    void erase_at(uint32_t idx) {
        PRECONDITION(idx < m_entries.size());
        PRECONDITION(m_entries[idx].active);
        m_entries[idx].value = V{};
        m_entries[idx].active = false;
        m_free.push_back(idx);
        on_erase(idx);
    }

    /// Replace value at handle, bump version. Marks the slot dirty with the
    /// supplied mask for each consumer (gated by subscription).
    void upsert(Handle h, V new_value, DirtyMask changed = DirtyMask{}) {
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        m_entries[h.idx].value = std::move(new_value);
        m_entries[h.idx].version = ++m_next_version;
        propagate_dirty(h.idx, changed);
    }

    /// In-place mutation; bumps version, then ORs `changed` into each
    /// consumer's per-slot dirty bits (gated by subscription).
    template <class Fn>
    void mutate(Handle h, DirtyMask changed, Fn&& fn) {
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        mutate_at_impl(h.idx, changed, std::forward<Fn>(fn));
    }

    /// Convenience overload when dirty tracking is disabled (monostate).
    template <class Fn, class M = DirtyMask,
              std::enable_if_t<std::is_same_v<M, std::monostate>, int> = 0>
    void mutate(Handle h, Fn&& fn) {
        mutate(h, DirtyMask{}, std::forward<Fn>(fn));
    }

    /// In-place mutation by raw slot index.
    template <class Fn>
    void mutate_at(uint32_t idx, DirtyMask changed, Fn&& fn) {
        PRECONDITION(idx < m_entries.size());
        PRECONDITION(m_entries[idx].active);
        mutate_at_impl(idx, changed, std::forward<Fn>(fn));
    }

    template <class Fn, class M = DirtyMask,
              std::enable_if_t<std::is_same_v<M, std::monostate>, int> = 0>
    void mutate_at(uint32_t idx, Fn&& fn) {
        mutate_at(idx, DirtyMask{}, std::forward<Fn>(fn));
    }

    // --- Index-based access ---

    const V& at(uint32_t idx) const {
        PRECONDITION(idx < m_entries.size());
        PRECONDITION(m_entries[idx].active);
        return m_entries[idx].value;
    }

    /// Mutable raw access. Bypasses dirty tracking -- use `mutate()` when
    /// consumers should be notified of the change. Exposed so composing
    /// containers (SlotMap) can implement non-const `for_each`.
    V& at_mut(uint32_t idx) {
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

    // --- Iteration ---

    /// Iterate active entries. Callback: fn(uint32_t idx, const V& value).
    template <class Fn>
    void for_each(Fn&& fn) const {
        for (uint32_t i = 0; i < m_entries.size(); ++i) {
            if (m_entries[i].active) fn(i, m_entries[i].value);
        }
    }

    /// Raw backing vector including tombstoned holes. Use for index-based
    /// GPU iteration where the slot index must match the buffer position.
    boost::span<const Entry> span_raw() const {
        return {m_entries.data(), m_entries.size()};
    }

    /// Number of currently-active entries.
    size_t size() const noexcept {
        return m_entries.size() - m_free.size();
    }

    /// Total vector capacity (live + tombstoned).
    size_t capacity() const noexcept {
        return m_entries.size();
    }

    void clear() {
        m_entries.clear();
        m_free.clear();
        for (auto& c : m_consumers) {
            c.per_slot_dirty.clear();
            c.erase_events.clear();
            c.inserts_pending = false;
            c.aggregate = DirtyMask{};
        }
        // m_next_version intentionally NOT reset -- monotonic across clears
    }

    // --- Dirty-mask tracking (no-op when DirtyMask is std::monostate) ---

    /// Register a new consumer. Newly registered consumers start fully
    /// dirty: aggregate = subscription, every active slot's per-slot mask
    /// is primed with `subscription`, and `inserts_pending = true`. The
    /// first drain emits on_update for every existing entry.
    ConsumerId register_consumer(DirtyMask subscription) {
        ConsumerState state;
        state.subscription = subscription;
        state.inserts_pending = true;
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

    /// True if any insert (or the initial priming from register_consumer)
    /// is pending for this consumer. Cleared on drain.
    bool any_inserts_for(ConsumerId id) const {
        PRECONDITION(id < m_consumers.size());
        return m_consumers[id].inserts_pending;
    }

    /// True if any erase events are queued for this consumer.
    bool any_erases_for(ConsumerId id) const {
        PRECONDITION(id < m_consumers.size());
        return !m_consumers[id].erase_events.empty();
    }

    /// Force `id`'s next drain to observe a full rebuild: primes every
    /// active slot's per-slot dirty with subscription, sets aggregate =
    /// subscription, and flips inserts_pending. Useful after an external
    /// reset (e.g. GPU resource teardown) where the map itself didn't
    /// change but downstream state did.
    void mark_consumer_fully_dirty(ConsumerId id) {
        PRECONDITION(id < m_consumers.size());
        auto& c = m_consumers[id];
        c.inserts_pending = true;
        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            if (c.per_slot_dirty.size() < m_entries.size()) {
                c.per_slot_dirty.resize(m_entries.size(), DirtyMask{});
            }
            for (size_t i = 0; i < m_entries.size(); ++i) {
                if (m_entries[i].active) {
                    detail::mask_or_assign(c.per_slot_dirty[i], c.subscription);
                }
            }
            c.aggregate = c.subscription;
        }
    }

    /// Drain erase events (via on_erase) and per-slot dirty bits intersecting
    /// `query` (via on_update). on_erase fires first so consumers may release
    /// per-slot state before any free-list reuse surfaces via on_update.
    ///
    /// Callbacks:
    ///   on_update(uint32_t slot_idx, const V& value)
    ///   on_erase(uint32_t slot_idx)
    ///
    /// Bits outside `query` are left intact for other drain calls or queries.
    template <class OnUpdate, class OnErase>
    void drain(ConsumerId id, DirtyMask query, OnUpdate&& on_update, OnErase&& on_erase) {
        PRECONDITION(id < m_consumers.size());
        auto& c = m_consumers[id];

        for (auto idx : c.erase_events) {
            on_erase(idx);
        }
        c.erase_events.clear();
        c.inserts_pending = false;

        if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
            DirtyMask remaining = DirtyMask{};
            for (uint32_t i = 0; i < c.per_slot_dirty.size(); ++i) {
                DirtyMask slot_bits = c.per_slot_dirty[i];
                if (!detail::mask_is_zero(detail::mask_and(slot_bits, query))) {
                    on_update(i, m_entries[i].value);
                    slot_bits = detail::mask_andnot(slot_bits, query);
                    c.per_slot_dirty[i] = slot_bits;
                }
                detail::mask_or_assign(remaining, slot_bits);
            }
            c.aggregate = remaining;
        } else {
            (void) query;
            (void) on_update;
        }
    }

    /// Read the consumer's subscription mask.
    DirtyMask subscription_for(ConsumerId id) const {
        PRECONDITION(id < m_consumers.size());
        return m_consumers[id].subscription;
    }

   private:
    struct ConsumerState {
        DirtyMask subscription{};
        std::vector<DirtyMask> per_slot_dirty;
        std::vector<uint32_t> erase_events;
        bool inserts_pending = false;
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

    void on_insert(uint32_t idx) {
        for (auto& c : m_consumers) {
            c.inserts_pending = true;
            if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
                if (c.per_slot_dirty.size() <= idx) {
                    c.per_slot_dirty.resize(idx + 1, DirtyMask{});
                }
                detail::mask_or_assign(c.per_slot_dirty[idx], c.subscription);
                detail::mask_or_assign(c.aggregate, c.subscription);
            }
        }
        (void) idx;
    }

    void on_erase(uint32_t idx) {
        for (auto& c : m_consumers) {
            if constexpr (detail::dirty_tracking_enabled_v<DirtyMask>) {
                if (idx < c.per_slot_dirty.size()) {
                    c.per_slot_dirty[idx] = DirtyMask{};
                }
                // aggregate may now be a (stale) superset; recomputed on
                // drain. any_dirty_for() still bounds-correct.
            }
            c.erase_events.push_back(idx);
        }
    }

    template <class Fn>
    void mutate_at_impl(uint32_t idx, DirtyMask changed, Fn&& fn) {
        std::forward<Fn>(fn)(m_entries[idx].value);
        m_entries[idx].version = ++m_next_version;
        propagate_dirty(idx, changed);
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
    std::vector<uint32_t> m_free;
    std::vector<ConsumerState> m_consumers;
    uint64_t m_next_version = 0;
};

}  // namespace pts::container
