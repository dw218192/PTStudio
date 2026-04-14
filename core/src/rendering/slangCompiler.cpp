// libslang backend — native only. WASM builds use EmbeddedCompiler exclusively
// and never include this translation unit's symbols.
#ifndef __EMSCRIPTEN__

#include <core/diagnostics.h>
#include <core/rendering/shaderc/shaderLoader.h>
#include <core/rendering/shaderc/slangRuntime.h>
#include <core/rendering/slangCompiler.h>
#include <slang-com-ptr.h>
#include <slang.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pts::rendering {

namespace {

// Disk-cache keys are NOT security-sensitive — only collision resistance among
// same-process inputs matters, and a std::size_t from boost::hash_combine over
// all relevant fields gives that at a fraction of the code of a cryptographic
// hash. Collisions would silently serve stale WGSL; format_version +
// defines_canon + source bytes + dep hashes together make a collision
// astronomically unlikely. If that ever becomes inadequate, swap a real
// cryptographic hash in here — the sidecar format is stable.
std::string hash_hex(std::size_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016zx", h);
    return std::string(buf, 16);
}

std::string read_file_bytes(const std::filesystem::path& p, std::error_code& ec) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool write_file_atomic(const std::filesystem::path& p, std::string_view contents) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    auto tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!f) return false;
    }
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        // Windows rename fails if the destination exists on some runtimes;
        // try remove + rename.
        std::filesystem::remove(p, ec);
        std::filesystem::rename(tmp, p, ec);
    }
    return !ec;
}

std::string canonical_defines(boost::span<const std::string_view> defines) {
    std::vector<std::string_view> sorted(defines.begin(), defines.end());
    std::sort(sorted.begin(), sorted.end());
    std::string out;
    for (auto& d : sorted) {
        out.append(d.data(), d.size());
        out.push_back('\n');
    }
    return out;
}

// Hash a file's bytes. Returns 0 on read failure (treated as "dep missing" —
// the computed cache key then won't match the stored meta, forcing recompile).
std::size_t hash_file(const std::filesystem::path& p) {
    std::error_code ec;
    auto bytes = read_file_bytes(p, ec);
    if (ec) return 0;
    return boost::hash<std::string>{}(bytes);
}

constexpr uint32_t k_cache_format_version = 1;
constexpr const char* k_target_profile = "wgsl";

// Per-variant result kept in memory for poll_dirty + revision tracking.
struct VariantResult {
    std::string cache_key;  // hash hex
    std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> deps;
    std::string wgsl;  // cached result
};

struct SourceEntry {
    std::mutex mutex;
    uint64_t revision = 1;
    // defines_canonical -> compiled variant
    std::unordered_map<std::string, VariantResult> variants;
};

}  // namespace

// ---------------------------------------------------------------------------
// SlangCompiler::Impl
// ---------------------------------------------------------------------------

struct SlangCompiler::Impl {
    const ShaderLoader* loader;
    std::shared_ptr<spdlog::logger> logger;
    std::filesystem::path cache_dir;
    std::filesystem::path workspace_root;
    std::filesystem::path search_path;
    IShaderCompiler* error_fallback;
    std::string slang_version;

    // Guards m_entries (map itself). Entry mutexes guard per-key state.
    mutable std::mutex entries_mutex;
    std::unordered_map<std::string, std::unique_ptr<SourceEntry>> entries;

    Slang::ComPtr<slang::IGlobalSession> global_session;
    std::mutex global_session_mutex;  // Slang global session is not thread-safe

    SourceEntry& get_or_create_entry(std::string_view source_key) {
        std::lock_guard<std::mutex> lock(entries_mutex);
        auto it = entries.find(std::string(source_key));
        if (it == entries.end()) {
            auto e = std::make_unique<SourceEntry>();
            auto [ins, _] = entries.emplace(std::string(source_key), std::move(e));
            it = ins;
        }
        return *it->second;
    }

    SourceEntry* find_entry(std::string_view source_key) const {
        std::lock_guard<std::mutex> lock(entries_mutex);
        auto it = entries.find(std::string(source_key));
        return it == entries.end() ? nullptr : it->second.get();
    }

    std::string cache_wgsl_path(const std::string& cache_key_hex) const {
        return (cache_dir / (cache_key_hex + ".wgsl")).string();
    }
    std::string cache_meta_path(const std::string& meta_key_hex) const {
        return (cache_dir / (meta_key_hex + ".meta")).string();
    }

    // Compose inputs that identify a compile: sorted defines + source bytes +
    // dep file hashes + slang_version + target_profile + format_version.
    std::string compute_cache_key(const std::string& source_bytes,
                                  const std::vector<std::filesystem::path>& deps,
                                  const std::string& defines_canon) const {
        std::size_t h = 0;
        boost::hash_combine(h, k_cache_format_version);
        boost::hash_combine(h, std::string_view(k_target_profile));
        boost::hash_combine(h, std::string_view(slang_version));
        boost::hash_combine(h, std::string_view(defines_canon));
        boost::hash_combine(h, std::string_view(source_bytes));
        for (auto& dep : deps) {
            boost::hash_combine(h, dep.string());
            boost::hash_combine(h, hash_file(dep));
        }
        return hash_hex(h);
    }

    // Meta key identifies a (source_key, defines) slot — stable across
    // recompiles. Value stored is the last cache_key + its dep list.
    std::string meta_key(std::string_view source_key, const std::string& defines_canon) const {
        std::size_t h = 0;
        boost::hash_combine(h, std::string_view("m1"));
        boost::hash_combine(h, source_key);
        boost::hash_combine(h, std::string_view(defines_canon));
        return hash_hex(h);
    }

    bool read_meta(const std::string& meta_key_hex, std::string& last_cache_key,
                   std::vector<std::filesystem::path>& deps) {
        std::ifstream f(cache_meta_path(meta_key_hex), std::ios::binary);
        if (!f) return false;
        std::getline(f, last_cache_key);
        if (last_cache_key.empty()) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) deps.emplace_back(line);
        }
        return true;
    }

    void write_meta(const std::string& meta_key_hex, const std::string& cache_key_hex,
                    const std::vector<std::filesystem::path>& deps) {
        std::ostringstream ss;
        ss << cache_key_hex << '\n';
        for (auto& dep : deps) ss << dep.string() << '\n';
        write_file_atomic(cache_meta_path(meta_key_hex), ss.str());
    }

    std::string fallback_or_panic(const ShaderKey& key, std::string_view diagnostics) {
        if (error_fallback) {
            logger->warn("Slang compile failed for '{}', falling back to embedded. {}", key.source,
                         diagnostics);
            return error_fallback->compile(key);
        }
        logger->error("Slang compile failed for '{}' and no fallback: {}", key.source, diagnostics);
        PANIC("Shader compile failed with no fallback");
    }

    std::string do_compile(const ShaderKey& key) {
        auto* loaded = loader->find(key.source);
        if (!loaded) {
            // Not a registered shader — fall back to embedded lookup.
            return error_fallback ? error_fallback->compile(key) : loader->load(key.source);
        }
        auto slang_path = workspace_root / loaded->slang_source;
        if (!std::filesystem::is_regular_file(slang_path)) {
            if (error_fallback) return error_fallback->compile(key);
            return loader->load(key.source);
        }

        auto defines_canon = canonical_defines(key.defines);
        auto& entry = get_or_create_entry(key.source);
        std::lock_guard<std::mutex> entry_lock(entry.mutex);

        // Fast path: in-memory cache hit.
        if (auto it = entry.variants.find(defines_canon); it != entry.variants.end()) {
            return it->second.wgsl;
        }

        // Try disk cache via meta sidecar.
        auto mk = meta_key(key.source, defines_canon);
        std::string last_cache_key;
        std::vector<std::filesystem::path> cached_deps;
        if (read_meta(mk, last_cache_key, cached_deps)) {
            std::error_code ec;
            auto src_bytes = read_file_bytes(slang_path, ec);
            if (!ec) {
                auto current_key = compute_cache_key(src_bytes, cached_deps, defines_canon);
                if (current_key == last_cache_key) {
                    auto wgsl = read_file_bytes(cache_wgsl_path(last_cache_key), ec);
                    if (!ec && !wgsl.empty()) {
                        VariantResult vr;
                        vr.cache_key = last_cache_key;
                        vr.wgsl = std::move(wgsl);
                        for (auto& d : cached_deps) {
                            auto mt = std::filesystem::last_write_time(d, ec);
                            vr.deps.emplace_back(d, ec ? std::filesystem::file_time_type{} : mt);
                        }
                        auto& slot = entry.variants[defines_canon];
                        slot = std::move(vr);
                        return slot.wgsl;
                    }
                }
            }
        }

        // Invoke libslang via the shared compile primitive (also used by
        // pts_shaderc build-time CLI).
        SlangCompileOutput out;
        {
            std::lock_guard<std::mutex> gs_lock(global_session_mutex);
            out = run_slang(global_session.get(), search_path, slang_path, loaded->entry_points,
                            key.defines);
        }
        if (!out.success) {
            return fallback_or_panic(key, out.diagnostics);
        }
        if (!out.diagnostics.empty()) {
            logger->warn("Slang '{}': {}", key.source, out.diagnostics);
        }

        std::error_code ec;
        auto src_bytes = read_file_bytes(slang_path, ec);
        if (ec) src_bytes.clear();
        auto cache_key = compute_cache_key(src_bytes, out.dependencies, defines_canon);
        write_file_atomic(cache_wgsl_path(cache_key), out.wgsl);
        write_meta(mk, cache_key, out.dependencies);

        VariantResult vr;
        vr.cache_key = std::move(cache_key);
        vr.wgsl = std::move(out.wgsl);
        for (auto& d : out.dependencies) {
            auto mt = std::filesystem::last_write_time(d, ec);
            vr.deps.emplace_back(d, ec ? std::filesystem::file_time_type{} : mt);
        }
        auto& slot = entry.variants[defines_canon];
        slot = std::move(vr);

        ++entry.revision;  // bump on every libslang compile so pipelines rebuild
        return slot.wgsl;
    }
};

// ---------------------------------------------------------------------------
// SlangCompiler
// ---------------------------------------------------------------------------

SlangCompiler::SlangCompiler(const ShaderLoader& loader, std::shared_ptr<spdlog::logger> logger,
                             std::filesystem::path cache_dir, std::filesystem::path workspace_root,
                             std::filesystem::path search_path, IShaderCompiler* error_fallback)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->loader = &loader;
    m_impl->logger = std::move(logger);
    m_impl->cache_dir = std::move(cache_dir);
    m_impl->workspace_root = std::move(workspace_root);
    m_impl->search_path = std::move(search_path);
    m_impl->error_fallback = error_fallback;

    std::error_code ec;
    std::filesystem::create_directories(m_impl->cache_dir, ec);

    auto hr = slang::createGlobalSession(m_impl->global_session.writeRef());
    INVARIANT_MSG(SLANG_SUCCEEDED(hr) && m_impl->global_session,
                  "Failed to create Slang global session");
    if (auto* tag = m_impl->global_session->getBuildTagString()) {
        m_impl->slang_version = tag;
    }
}

SlangCompiler::~SlangCompiler() = default;

std::string SlangCompiler::compile(const ShaderKey& key) {
    return m_impl->do_compile(key);
}

std::vector<std::string> SlangCompiler::poll_dirty() {
    std::vector<std::string> dirty;
    // Snapshot keys under entries_mutex; check each entry under its own lock.
    std::vector<std::string> keys;
    {
        std::lock_guard<std::mutex> lock(m_impl->entries_mutex);
        keys.reserve(m_impl->entries.size());
        for (auto& [k, _] : m_impl->entries) keys.push_back(k);
    }
    for (auto& key : keys) {
        auto* entry = m_impl->find_entry(key);
        if (!entry) continue;
        std::lock_guard<std::mutex> lock(entry->mutex);
        bool is_dirty = false;
        for (auto& [defines_canon, variant] : entry->variants) {
            for (auto& [dep_path, last_mtime] : variant.deps) {
                std::error_code ec;
                auto mt = std::filesystem::last_write_time(dep_path, ec);
                if (ec) continue;
                if (mt != last_mtime) {
                    is_dirty = true;
                    break;
                }
            }
            if (is_dirty) break;
        }
        if (is_dirty) {
            // Drop cached variants; next compile() will rebuild from current source.
            entry->variants.clear();
            ++entry->revision;
            dirty.push_back(key);
        }
    }
    if (!dirty.empty()) {
        m_impl->logger->info("Shader change detected: {} source(s) dirty", dirty.size());
    }
    return dirty;
}

uint64_t SlangCompiler::source_revision(std::string_view source_key) const {
    auto* entry = m_impl->find_entry(source_key);
    if (!entry) return 1;
    std::lock_guard<std::mutex> lock(entry->mutex);
    return entry->revision;
}

void SlangCompiler::invalidate(std::string_view source_key) {
    auto& entry = m_impl->get_or_create_entry(source_key);
    std::lock_guard<std::mutex> lock(entry.mutex);
    entry.variants.clear();
    ++entry.revision;
}

}  // namespace pts::rendering

#endif  // !__EMSCRIPTEN__
