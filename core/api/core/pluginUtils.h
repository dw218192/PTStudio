#pragma once
#include <cstring>
#include <type_traits>

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

}  // namespace detail

/**
 * Base lifecycle interface for C++ plugin implementations.
 * Plugins can inherit from this for convenience.
 */
struct IPlugin {
    explicit IPlugin(PtsHostApi* host_api) : m_host_api(host_api) {
    }
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

   protected:
    PtsHostApi* host_api() const {
        return m_host_api;
    }

   private:
    PtsHostApi* m_host_api = nullptr;
};

}  // namespace pts

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
                    return new PluginClass(host_api);                               \
                } catch (...) {                                                     \
                    return nullptr;                                                 \
                }                                                                   \
            },                                                                      \
            [](PluginHandle p) { delete static_cast<PluginClass*>(p); },            \
            [](PluginHandle p) { return static_cast<PluginClass*>(p)->on_load(); }, \
            [](PluginHandle p) { static_cast<PluginClass*>(p)->on_unload(); },      \
            [](PluginHandle p, const char* iid) {                                   \
                return pts::detail::query_interface_impl<PluginClass>(p, iid);      \
            }};                                                                     \
        return &desc;                                                               \
    }                                                                               \
    }

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
 * Type-safe wrapper for querying plugin interfaces.
 *
 * Usage:
 *   auto* iface = query_plugin_interface<MyInterfaceV1>(plugin_handle, descriptor,
 * "my_interface_v1"); if (iface) { iface->do_something();
 *   }
 */
template <typename InterfaceType>
inline InterfaceType* query_plugin_interface(PluginHandle handle, const PtsPluginDescriptor* desc,
                                             const char* interface_id) {
    if (!handle || !desc || !desc->query_interface) {
        return nullptr;
    }
    return static_cast<InterfaceType*>(desc->query_interface(handle, interface_id));
}

/**
 * Compile-time interface ID wrapper for type safety.
 *
 * Usage:
 *   constexpr auto MyInterfaceID = PluginInterfaceID<MyInterfaceV1>("my_interface_v1");
 *   auto* iface = MyInterfaceID.query(plugin_handle, descriptor);
 */
template <typename InterfaceType>
struct PluginInterfaceID {
    const char* id;

    constexpr explicit PluginInterfaceID(const char* interface_id) : id(interface_id) {
    }

    InterfaceType* query(PluginHandle handle, const PtsPluginDescriptor* desc) const {
        return query_plugin_interface<InterfaceType>(handle, desc, id);
    }
};

}  // namespace pts