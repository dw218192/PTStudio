#pragma once

#include <core/diagnostics.h>

#include <boost/container/small_vector.hpp>
#include <boost/container_hash/hash.hpp>
#include <boost/core/span.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pts::cache {

/// Generic dependency-tracked cache.
///
/// Keyed map of Values where each entry carries a monotonic `version` and a
/// snapshot of the dep version vector it was built under. `get_or_build`
/// rebuilds when any dep changed or when `invalidate(key)` was called.
///
/// Entries are node-wrapped via unique_ptr so references returned by
/// `get_or_build` remain stable across subsequent inserts/rebuilds of other
/// entries (Boost flat_map may rehash and move pairs, but the pointed-to
/// Entry stays put).
template <typename Key, typename Value, typename Hash = boost::hash<Key>,
          typename Eq = std::equal_to<Key>>
class DepTrackedCache {
   public:
    using Span = boost::span<const uint64_t>;

    DepTrackedCache() = default;
    DepTrackedCache(const DepTrackedCache&) = delete;
    DepTrackedCache& operator=(const DepTrackedCache&) = delete;
    DepTrackedCache(DepTrackedCache&&) = default;
    DepTrackedCache& operator=(DepTrackedCache&&) = default;

    /// Build on miss or when `current_deps` differs from the entry's snapshot
    /// or when `invalidate(key)` was called. Returns stable reference.
    template <typename K2, typename BuildFn>
    const Value& get_or_build(const K2& key, Span current_deps, BuildFn&& build) {
        auto it = m_entries.find(key);
        if (it == m_entries.end()) {
            auto entry = std::make_unique<Entry>();
            entry->value = std::forward<BuildFn>(build)();
            entry->version = ++m_next_version;
            entry->deps_snapshot.assign(current_deps.begin(), current_deps.end());
            auto [ins_it, _] = m_entries.emplace(Key(key), std::move(entry));
            return ins_it->second->value;
        }
        Entry& e = *it->second;
        if (!e.forced_dirty && deps_match(e.deps_snapshot, current_deps)) {
            return e.value;
        }
        e.value = std::forward<BuildFn>(build)();
        e.version = ++m_next_version;
        e.deps_snapshot.assign(current_deps.begin(), current_deps.end());
        e.forced_dirty = false;
        return e.value;
    }

    /// Build on miss or dep/forced-dirty mismatch. Calls `on_replace(old_value)`
    /// BEFORE writing the new value; useful for releasing GPU handles before
    /// overwriting. Returns the resulting stable reference.
    template <typename K2, typename BuildFn, typename OnReplaceFn>
    const Value& get_or_build_with_replace(const K2& key, Span current_deps, BuildFn&& build,
                                           OnReplaceFn&& on_replace) {
        auto it = m_entries.find(key);
        if (it == m_entries.end()) {
            auto entry = std::make_unique<Entry>();
            entry->value = std::forward<BuildFn>(build)();
            entry->version = ++m_next_version;
            entry->deps_snapshot.assign(current_deps.begin(), current_deps.end());
            auto [ins_it, _] = m_entries.emplace(Key(key), std::move(entry));
            return ins_it->second->value;
        }
        Entry& e = *it->second;
        if (!e.forced_dirty && deps_match(e.deps_snapshot, current_deps)) {
            return e.value;
        }
        std::forward<OnReplaceFn>(on_replace)(e.value);
        e.value = std::forward<BuildFn>(build)();
        e.version = ++m_next_version;
        e.deps_snapshot.assign(current_deps.begin(), current_deps.end());
        e.forced_dirty = false;
        return e.value;
    }

    /// Version of the entry for `key`, or 0 if not yet built.
    template <typename K2>
    uint64_t version(const K2& key) const {
        auto it = m_entries.find(key);
        if (it == m_entries.end()) return 0;
        return it->second->version;
    }

    /// Pointer to the value, or nullptr if not built. Stable across inserts.
    template <typename K2>
    const Value* find(const K2& key) const {
        auto it = m_entries.find(key);
        if (it == m_entries.end()) return nullptr;
        return &it->second->value;
    }
    template <typename K2>
    Value* find(const K2& key) {
        auto it = m_entries.find(key);
        if (it == m_entries.end()) return nullptr;
        return &it->second->value;
    }

    template <typename K2>
    bool contains(const K2& key) const {
        return m_entries.find(key) != m_entries.end();
    }

    /// Force the next `get_or_build` for this key to rebuild, even if deps
    /// match. No-op if the key isn't present.
    template <typename K2>
    void invalidate(const K2& key) {
        auto it = m_entries.find(key);
        if (it == m_entries.end()) return;
        it->second->forced_dirty = true;
    }

    /// Force all entries to rebuild on next get_or_build.
    void invalidate_all() {
        for (auto& [_, entry] : m_entries) {
            entry->forced_dirty = true;
        }
    }

    /// Drop the entry. Next build starts with a fresh version tag.
    /// Returns a pointer to the unique_ptr wrapping the old value so the caller
    /// may extract and release GPU handles before destruction.
    std::unique_ptr<Value> erase_extract(const Key& key) {
        // Not directly supported -- Entry owns the value, not the Value alone.
        // Keep this as a placeholder; callers can fetch via find() then erase().
        UNUSED(key);
        return nullptr;
    }

    /// Drop the entry.
    template <typename K2>
    void erase(const K2& key) {
        auto it = m_entries.find(key);
        if (it != m_entries.end()) m_entries.erase(it);
    }

    /// Iterate entries. Callback signature: (const Key&, Value&) or (const Key&, const Value&).
    template <typename Fn>
    void for_each(Fn&& fn) {
        for (auto& [k, entry] : m_entries) {
            fn(k, entry->value);
        }
    }
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (const auto& [k, entry] : m_entries) {
            fn(k, entry->value);
        }
    }

    void clear() {
        m_entries.clear();
    }

    size_t size() const noexcept {
        return m_entries.size();
    }

    bool empty() const noexcept {
        return m_entries.empty();
    }

   private:
    struct Entry {
        Value value{};
        uint64_t version = 0;
        boost::container::small_vector<uint64_t, 4> deps_snapshot;
        bool forced_dirty = false;
    };

    template <typename V>
    static bool deps_match(const V& snapshot, Span current) {
        if (snapshot.size() != current.size()) return false;
        for (size_t i = 0; i < current.size(); ++i) {
            if (snapshot[i] != current[i]) return false;
        }
        return true;
    }

    boost::unordered_flat_map<Key, std::unique_ptr<Entry>, Hash, Eq> m_entries;
    uint64_t m_next_version = 0;
};

}  // namespace pts::cache
