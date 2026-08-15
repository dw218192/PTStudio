#pragma once

#include <core/container/slotArray.h>
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

/// Keyed slot map: SlotArray<V, DirtyMask> plus a K -> slot-index lookup.
///
/// Indices are stable (provided by the underlying SlotArray's free-list),
/// so handles and raw cross-references survive unrelated erases and vector
/// reallocation. Dirty tracking, erase event queues, and the `drain()` API
/// all forward to the composed SlotArray.
///
/// K must be LessComparable. Compare defaults to std::less<K>; pass
/// std::less<> for transparent (heterogeneous) lookup on string-like keys.
template <class K, class V, class DirtyMask = std::monostate, class Compare = std::less<K>>
class SlotMap {
   public:
    using Array = SlotArray<V, DirtyMask>;
    using Entry = typename Array::Entry;
    using ConsumerId = typename Array::ConsumerId;

    struct Handle {
        const SlotMap* cache = nullptr;
        uint32_t idx = UINT32_MAX;

        const V& operator*() const {
            PRECONDITION(cache);
            return cache->m_storage.at(idx);
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

    /// Insert a new entry keyed by `key`. Asserts key is not already present.
    Handle insert(K key, V value) {
        PRECONDITION_MSG(!contains(key), "SlotMap::insert: duplicate key");
        auto sh = m_storage.insert(std::move(value));
        m_index.emplace(std::move(key), sh.idx);
        return Handle{this, sh.idx};
    }

    /// Replace value at handle (monostate API).
    void upsert(Handle h, V new_value, DirtyMask changed = DirtyMask{}) {
        PRECONDITION(h.cache == this);
        m_storage.upsert(typename Array::Handle{h.idx}, std::move(new_value), changed);
    }

    template <class Fn>
    void mutate(Handle h, DirtyMask changed, Fn&& fn) {
        PRECONDITION(h.cache == this);
        m_storage.mutate(typename Array::Handle{h.idx}, changed, std::forward<Fn>(fn));
    }

    template <class Fn, class M = DirtyMask,
              std::enable_if_t<std::is_same_v<M, std::monostate>, int> = 0>
    void mutate(Handle h, Fn&& fn) {
        mutate(h, DirtyMask{}, std::forward<Fn>(fn));
    }

    template <class Fn>
    void mutate_at(uint32_t idx, DirtyMask changed, Fn&& fn) {
        m_storage.mutate_at(idx, changed, std::forward<Fn>(fn));
    }

    template <class Fn, class M = DirtyMask,
              std::enable_if_t<std::is_same_v<M, std::monostate>, int> = 0>
    void mutate_at(uint32_t idx, Fn&& fn) {
        m_storage.mutate_at(idx, std::forward<Fn>(fn));
    }

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

    template <class K2>
    void erase(const K2& key) {
        auto it = m_index.find(key);
        if (it == m_index.end()) return;
        auto idx = it->second;
        m_index.erase(it);
        m_storage.erase_at(idx);
    }

    uint64_t version(Handle h) const {
        PRECONDITION(h.cache == this);
        return m_storage.version_at(h.idx);
    }

    // --- Index-based / raw access (forwards to SlotArray) ---

    const V& at(uint32_t idx) const {
        return m_storage.at(idx);
    }
    bool active_at(uint32_t idx) const {
        return m_storage.active_at(idx);
    }
    uint64_t version_at(uint32_t idx) const {
        return m_storage.version_at(idx);
    }
    boost::span<const Entry> span_raw() const {
        return m_storage.span_raw();
    }

    /// Live entry count (number of unique keys).
    size_t size() const noexcept {
        return m_index.size();
    }

    /// Total slot capacity (live + tombstoned holes).
    size_t capacity() const noexcept {
        return m_storage.capacity();
    }

    void clear() {
        m_storage.clear();
        m_index.clear();
    }

    /// Iterate active entries by key. Callback: fn(const K& key, V& value).
    /// Mutations through the callback bypass dirty tracking -- call
    /// `mutate()` inside the loop if consumers must observe the change.
    template <class Fn>
    void for_each(Fn&& fn) {
        for (auto& [key, idx] : m_index) {
            fn(key, m_storage.at_mut(idx));
        }
    }

    /// Iterate active entries by key (const). Callback: fn(const K&, const V&).
    template <class Fn>
    void for_each(Fn&& fn) const {
        for (const auto& [key, idx] : m_index) {
            fn(key, m_storage.at(idx));
        }
    }

    // --- Dirty-mask tracking (forwards to SlotArray) ---

    ConsumerId register_consumer(DirtyMask subscription) {
        return m_storage.register_consumer(subscription);
    }
    bool any_dirty_for(ConsumerId id, DirtyMask query) const {
        return m_storage.any_dirty_for(id, query);
    }
    bool any_inserts_for(ConsumerId id) const {
        return m_storage.any_inserts_for(id);
    }
    bool any_erases_for(ConsumerId id) const {
        return m_storage.any_erases_for(id);
    }
    void mark_consumer_fully_dirty(ConsumerId id) {
        m_storage.mark_consumer_fully_dirty(id);
    }
    template <class OnUpdate, class OnErase>
    void drain(ConsumerId id, DirtyMask query, OnUpdate&& on_update, OnErase&& on_erase) {
        m_storage.drain(id, query, std::forward<OnUpdate>(on_update),
                        std::forward<OnErase>(on_erase));
    }
    DirtyMask subscription_for(ConsumerId id) const {
        return m_storage.subscription_for(id);
    }

   private:
    Array m_storage;
    boost::container::flat_map<K, uint32_t, Compare> m_index;
};

}  // namespace pts::container
