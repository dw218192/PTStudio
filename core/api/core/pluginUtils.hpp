#pragma once
#include <cstring>
#include <type_traits>
#include <utility>

#include "loggingUtils.hpp"
#include "plugin.h"

/**
 * This header contains inlined utility & macros & classes to help implement a plugin.
 */

namespace pts {
namespace detail {
/**
 * FNV-1a 64-bit hash function. Used to generate a unique identifier for a plugin interface.
 */
constexpr uint64_t fnv1a64(const char* s) {
    uint64_t h = 14695981039346656037ull;
    for (; *s; ++s) {
        h ^= static_cast<uint8_t>(*s);
        h *= 1099511628211ull;
    }
    return h;
}

// ============================================================================
// Type List Metaprogramming Infrastructure
// ============================================================================

/**
 * Empty type list terminator.
 */
struct TypeListEnd {};

/**
 * Type list node.
 */
template <typename Head, typename Tail = TypeListEnd>
struct TypeList {
    using head = Head;
    using tail = Tail;
};

/**
 * Interface registration entry - associates an interface ID with a getter function.
 * The getter function takes no parameters - plugin instance is automatically set in thread-local
 * storage.
 */
template <const char* InterfaceId, typename InterfaceType, typename PluginClass,
          InterfaceType* (*GetterFunc)()>
struct InterfaceEntry {
    static constexpr const char* id = InterfaceId;
    using interface_type = InterfaceType;
    static constexpr auto getter = GetterFunc;

    static void* query(const char* iid, void* plugin_handle) {
        if (std::strcmp(iid, id) == 0) {
            // Set the plugin instance in thread-local storage before calling getter
            PluginInstanceStorage<PluginClass>::instance = static_cast<PluginClass*>(plugin_handle);
            return getter();
        }
        return nullptr;
    }
};

/**
 * Recursive interface query through type list.
 * Passes the plugin handle through to allow OOP-style method binding.
 */
template <typename InterfaceList>
struct InterfaceQuery {
    static void* query(const char* iid, void* plugin_handle) {
        // Try current interface
        void* result = InterfaceList::head::query(iid, plugin_handle);
        if (result) {
            return result;
        }
        // Recurse to next interface
        return InterfaceQuery<typename InterfaceList::tail>::query(iid, plugin_handle);
    }
};

/**
 * Base case - no more interfaces to check.
 */
template <>
struct InterfaceQuery<TypeListEnd> {
    static void* query(const char* /*iid*/, void* /*plugin_handle*/) {
        return nullptr;
    }
};

/**
 * Base class for plugin interface tables.
 * Plugins specialize this template to define their interface list.
 */
template <typename PluginClass>
struct PluginInterfaceTable {
    using interface_list = TypeListEnd;  // Default: no interfaces
};

/**
 * Query interface implementation - uses metaprogramming to dispatch.
 * Passes the plugin instance to allow OOP-style method binding.
 */
template <typename PluginClass>
inline void* query_interface_impl(PluginHandle instance, const char* iid) {
    if (!instance || !iid) {
        return nullptr;
    }

    // Use the compile-time interface table to query, passing the plugin handle
    return InterfaceQuery<typename PluginInterfaceTable<PluginClass>::interface_list>::query(
        iid, instance);
}

template <typename InterfaceList>
struct InterfaceReporter {
    static void log(const PluginLogger& logger) noexcept {
        logger.log_info("  - {}", InterfaceList::head::id);
        InterfaceReporter<typename InterfaceList::tail>::log(logger);
    }
};

template <>
struct InterfaceReporter<TypeListEnd> {
    static void log(const PluginLogger& /*logger*/) noexcept {
    }
};

template <typename PluginClass>
inline void log_plugin_started(const PluginLogger& logger) noexcept {
    logger.log_info("Plugin started");
    using interface_list = typename PluginInterfaceTable<PluginClass>::interface_list;
    if constexpr (std::is_same_v<interface_list, TypeListEnd>) {
        logger.log_info("Exported interfaces: none");
    } else {
        logger.log_info("Exported interfaces:");
        InterfaceReporter<interface_list>::log(logger);
    }
}

inline void log_plugin_shutdown(const PluginLogger& logger) noexcept {
    logger.log_info("Plugin shutting down");
}

}  // namespace detail

// Registers static plugins into the host registry (no-op for dynamic builds).
void register_static_plugin(const PtsPluginDescriptor* descriptor) noexcept;

/**
 * Base lifecycle interface for C++ plugin implementations.
 * Plugins can inherit from this for convenience.
 */
struct IPlugin {
    IPlugin() = default;
    virtual ~IPlugin() = default;

    /**
     * Called after the plugin is loaded.
     * Return true if the plugin initialization was successful, false otherwise.
     * A plugin that fails to initialize will be unloaded.
     */
    virtual bool on_load() = 0;

    /**
     * Called before the plugin is unloaded.
     */
    virtual void on_unload() = 0;

    PluginLogger& logger() noexcept {
        return m_logger;
    }

    const PluginLogger& logger() const noexcept {
        return m_logger;
    }

    PtsHostApi* host_api() const noexcept {
        return m_host_api;
    }

   private:
    // hidden friend to hack access to initialize()
    template <typename PluginClass>
    friend bool initialize_plugin(PluginClass* plugin, PtsHostApi* host_api,
                                  PluginLogger logger) noexcept {
        if (!plugin || !host_api || !logger.is_valid()) {
            return false;
        }
        plugin->initialize(host_api, std::move(logger));
        return true;
    }

    void initialize(PtsHostApi* host_api, PluginLogger logger) noexcept {
        m_host_api = host_api;
        m_logger = std::move(logger);
    }
    PtsHostApi* m_host_api = nullptr;
    PluginLogger m_logger;
};

}  // namespace pts

#if defined(PTS_STATIC_PLUGINS)
#define PTS_PLUGIN_REGISTER_STATIC(PluginClass)                               \
    namespace {                                                               \
    struct PluginClass##StaticRegistrar {                                     \
        PluginClass##StaticRegistrar() {                                      \
            ::pts::register_static_plugin(::pts_plugin_get_desc());           \
        }                                                                     \
    };                                                                        \
    static PluginClass##StaticRegistrar PluginClass##StaticRegistrarInstance; \
    }  // namespace
#else
#define PTS_PLUGIN_REGISTER_STATIC(PluginClass)
#endif

/**
 * Defines a plugin. This should be put in exactly one translation unit of the plugin.
 *
 * @param PluginClass The C++ class implementing the plugin (must inherit from pts::IPlugin)
 * @param Kind The plugin kind (PTS_PLUGIN_KIND_SUBSYSTEM or PTS_PLUGIN_KIND_RENDERER)
 * @param Id Unique plugin identifier string
 * @param Name Human-readable display name
 * @param Version Version string
 */
#define PTS_PLUGIN_DEFINE(PluginClass, Kind, Id, Name, Version)                     \
    extern "C" {                                                                    \
    PTS_PLUGIN_EXPORT const PtsPluginDescriptor* pts_plugin_get_desc(void) {        \
        static PtsPluginDescriptor desc = {                                         \
            PTS_PLUGIN_API_VERSION,                                                 \
            sizeof(PtsPluginDescriptor),                                            \
            Kind,                                                                   \
            Id,                                                                     \
            Name,                                                                   \
            Version,                                                                \
            [](PtsHostApi* host_api) -> PluginHandle {                              \
                try {                                                               \
                    auto logger = pts::make_logger(host_api, Id);                   \
                    auto* plugin = new PluginClass();                               \
                    if (!initialize_plugin(plugin, host_api, std::move(logger))) {  \
                        delete plugin;                                              \
                        return nullptr;                                             \
                    }                                                               \
                    return plugin;                                                  \
                } catch (...) {                                                     \
                    return nullptr;                                                 \
                }                                                                   \
            },                                                                      \
            [](PluginHandle p) { delete static_cast<PluginClass*>(p); },            \
            [](PluginHandle p) {                                                    \
                auto* plugin = static_cast<PluginClass*>(p);                        \
                bool ok = plugin && plugin->on_load();                              \
                if (ok) {                                                           \
                    pts::detail::log_plugin_started<PluginClass>(plugin->logger()); \
                }                                                                   \
                return ok;                                                          \
            },                                                                      \
            [](PluginHandle p) {                                                    \
                auto* plugin = static_cast<PluginClass*>(p);                        \
                if (plugin) {                                                       \
                    pts::detail::log_plugin_shutdown(plugin->logger());             \
                    plugin->on_unload();                                            \
                }                                                                   \
            },                                                                      \
            [](PluginHandle p, const char* iid) {                                   \
                return pts::detail::query_interface_impl<PluginClass>(p, iid);      \
            }};                                                                     \
        return &desc;                                                               \
    }                                                                               \
    }                                                                               \
    PTS_PLUGIN_REGISTER_STATIC(PluginClass)

// ============================================================================
// Metaprogramming-based Interface Registration Macros
// ============================================================================

/**
 * Declares an interface ID constant.
 * Place at file scope (outside any classes).
 *
 * Example:
 *   PTS_INTERFACE_ID(kMyInterfaceId, "my.interface.v1");
 */
#define PTS_INTERFACE_ID(VarName, StringLiteral)        \
    namespace {                                         \
    constexpr char VarName##_storage[] = StringLiteral; \
    }                                                   \
    static constexpr const char* VarName = VarName##_storage;

/**
 * Registers plugin interfaces.
 * Each interface uses PTS_INTERFACE macro within the block.
 *
 * Example with 2 interfaces:
 *   PTS_INTERFACE_ID(kInterfaceId1, "plugin.interface.v1");
 *   PTS_INTERFACE_ID(kInterfaceId2, "plugin.math.v1");
 *
 *   PTS_PLUGIN_INTERFACES(MyPlugin,
 *       PTS_INTERFACE(MyPlugin, kInterfaceId1, MyInterfaceV1, MyPlugin::get_interface1)
 *       PTS_INTERFACE(MyPlugin, kInterfaceId2, MyInterfaceV2, MyPlugin::get_interface2)
 *   )
 *
 * The getter functions should have signature: InterfaceType* (*)()
 * The plugin instance is automatically available via thread-local storage.
 */

#define PTS_INTERFACE(PluginClass, Id, Type, Getter) \
    ::pts::detail::TypeList < ::pts::detail::InterfaceEntry<Id, Type, PluginClass, Getter>,

#define PTS_PLUGIN_INTERFACES(PluginClass, ...)                            \
    namespace pts {                                                        \
    namespace detail {                                                     \
    template <>                                                            \
    struct PluginInterfaceTable<PluginClass> {                             \
        using interface_list = __VA_ARGS__ ::pts::detail::TypeListEnd >> ; \
    };                                                                     \
    }                                                                      \
    }

// ============================================================================
// OOP-Style Method Binding Helpers (Auto-Generated Wrappers)
// ============================================================================

/**
 * Automatic method binding that generates clean wrapper functions.
 * Interface functions don't need void* plugin_handle parameters.
 * The plugin instance is automatically set in thread-local storage during query_interface.
 *
 * Usage - just define your methods normally:
 *   class MyPlugin : public pts::IPlugin {
 *       const char* get_greeting() { return "Hello"; }
 *       int add(int a, int b) { return a + b; }
 *
 *       static MyInterfaceV1* get_my_interface() {
 *           static MyInterfaceV1 table = {
 *               1, // version
 *               PTS_METHOD(MyPlugin, get_greeting, const char*),
 *               PTS_METHOD(MyPlugin, add, int, int, int)
 *           };
 *           return &table;
 *       }
 *   };
 *
 * The wrapper functions automatically access the plugin instance from thread-local storage.
 */

namespace pts {
namespace detail {

// Plugin instance storage for each plugin class (used by interface wrappers)
// Automatically set during interface query before calling the getter function
template <typename PluginClass>
struct PluginInstanceStorage {
    static thread_local PluginClass* instance;
};

template <typename PluginClass>
thread_local PluginClass* PluginInstanceStorage<PluginClass>::instance = nullptr;

// Helper template to generate wrapper functions that access the stored plugin instance
template <typename PluginClass, typename RetType, typename... Args>
struct MethodBinder {
    using MethodPtr = RetType (PluginClass::*)(Args...);

    template <MethodPtr Method>
    static RetType wrapper(Args... args) {
        // Access the stored plugin instance
        auto* plugin = PluginInstanceStorage<PluginClass>::instance;
        if (plugin) {
            return (plugin->*Method)(args...);
        }
        // Fallback for when instance is not set (shouldn't happen)
        if constexpr (std::is_default_constructible_v<RetType>) {
            return RetType{};
        }
    }
};
}  // namespace detail
}  // namespace pts

// Auto-generate wrapper - no manual binding needed!
#define PTS_METHOD(PluginClass, MethodName, RetType, ...) \
    &::pts::detail::MethodBinder<PluginClass, RetType,    \
                                 ##__VA_ARGS__>::wrapper<&PluginClass::MethodName>

// ============================================================================
// Host-side interface query helpers (C++)
// ============================================================================

namespace pts {

/**
 * C++ convenience wrapper over the host API interface query functions.
 */
class PluginInterfaceQuery final {
   public:
    PluginInterfaceQuery() = default;

    explicit PluginInterfaceQuery(PtsHostApi* host_api) noexcept : m_host_api(host_api) {
    }

    [[nodiscard]] auto is_valid() const noexcept -> bool {
        return m_host_api && m_host_api->query_interface;
    }

    template <typename InterfaceType>
    InterfaceType* query(PluginHandle handle, const char* interface_id) const noexcept {
        if (!handle || !interface_id || !m_host_api || !m_host_api->query_interface) {
            return nullptr;
        }
        return static_cast<InterfaceType*>(m_host_api->query_interface(handle, interface_id));
    }

   private:
    PtsHostApi* m_host_api = nullptr;
};

[[nodiscard]] inline auto make_interface_query(PtsHostApi* host_api) noexcept
    -> PluginInterfaceQuery {
    return PluginInterfaceQuery(host_api);
}

}  // namespace pts