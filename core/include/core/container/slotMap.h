#pragma once

#include <core/diagnostics.h>

#include <boost/container/flat_map.hpp>
#include <boost/core/span.hpp>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace pts::container {

/// Dense slot-map with stable indices and fat-pointer handles.
///
/// Backing storage is a flat vector of Entry structs. Erase tombstones the
/// slot and pushes it onto a free-list for reuse -- indices are never
/// shifted, so handles (and raw indices stored in cross-references like
/// ObjectData::mesh_index) survive unrelated erases and vector reallocation.
///
/// K must be LessComparable. Compare defaults to std::less<K>; pass
/// std::less<> for transparent (heterogeneous) lookup on string-like keys.
template <class K, class V, class Compare = std::less<K>>
class SlotMap {
   public:
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

    /// Insert a new entry. Asserts key is not already present.
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
        }
        m_index.emplace(std::move(key), idx);
        return Handle{this, idx};
    }

    /// Replace value at handle, bump version (globally monotonic).
    void upsert(Handle h, V new_value) {
        PRECONDITION(h.cache == this);
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        m_entries[h.idx].value = std::move(new_value);
        m_entries[h.idx].version = ++m_next_version;
    }

    /// In-place mutation; bumps version (globally monotonic) after fn returns.
    template <class Fn>
    void mutate(Handle h, Fn&& fn) {
        PRECONDITION(h.cache == this);
        PRECONDITION(h.idx < m_entries.size());
        PRECONDITION(m_entries[h.idx].active);
        std::forward<Fn>(fn)(m_entries[h.idx].value);
        m_entries[h.idx].version = ++m_next_version;
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
    /// RAII resources (GPU handles etc.) immediately.
    template <class K2>
    void erase(const K2& key) {
        auto it = m_index.find(key);
        if (it == m_index.end()) return;
        auto idx = it->second;
        m_entries[idx].value = V{};
        m_entries[idx].active = false;
        m_free.push_back(idx);
        m_index.erase(it);
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
    void mutate_at(uint32_t idx, Fn&& fn) {
        PRECONDITION(idx < m_entries.size());
        PRECONDITION(m_entries[idx].active);
        std::forward<Fn>(fn)(m_entries[idx].value);
        m_entries[idx].version = ++m_next_version;
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
        // m_next_version intentionally NOT reset -- monotonic across clears
    }

   private:
    std::vector<Entry> m_entries;
    boost::container::flat_map<K, uint32_t, Compare> m_index;
    std::vector<uint32_t> m_free;
    uint64_t m_next_version = 0;
};

}  // namespace pts::container
