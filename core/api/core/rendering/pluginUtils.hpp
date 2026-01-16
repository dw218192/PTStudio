#pragma once

#include "../pluginUtils.hpp"
#include "plugin.h"

namespace pts {
/**
 * C++ convenience interface for renderer plugins.
 * Implement these methods and use DEFINE_RENDERER_PLUGIN(...) to export.
 */
struct IRenderer {
    virtual ~IRenderer() = default;
    virtual void build_graph(const PtsHostApi* host, const PtsFrameParams* frame,
                             const PtsViewParams* view, const PtsFrameIO* io) = 0;
    virtual void on_resize(uint32_t w, uint32_t h) = 0;
    virtual void get_debug_outputs(PtsSpan* list_blob) = 0;
    virtual void set_settings_blob(PtsSpan blob) = 0;
    virtual void get_settings_schema(PtsSpan* schema_blob) = 0;
};
}  // namespace pts

PTS_INTERFACE_ID(kRendererInterfaceId, RENDERER_PLUGIN_INTERFACE_V1_ID);

/**
 * Defines a renderer plugin with the render graph interface.
 * Place this in exactly one translation unit.
 */
#define DEFINE_RENDERER_PLUGIN(PluginClass, Id, Name, Version)                                   \
    namespace {                                                                                  \
    inline RendererPluginInterfaceV1* get_renderer_interface() {                                 \
        static RendererPluginInterfaceV1 interface_table = {                                     \
            PTS_METHOD(PluginClass, build_graph, void, const PtsHostApi*, const PtsFrameParams*, \
                       const PtsViewParams*, const PtsFrameIO*),                                 \
            PTS_METHOD(PluginClass, on_resize, void, uint32_t, uint32_t),                        \
            PTS_METHOD(PluginClass, get_debug_outputs, void, PtsSpan*),                          \
            PTS_METHOD(PluginClass, set_settings_blob, void, PtsSpan),                           \
            PTS_METHOD(PluginClass, get_settings_schema, void, PtsSpan*)};                       \
        return &interface_table;                                                                 \
    }                                                                                            \
    }                                                                                            \
    PTS_PLUGIN_INTERFACES(                                                                       \
        PluginClass, PTS_INTERFACE(PluginClass, kRendererInterfaceId, RendererPluginInterfaceV1, \
                                   get_renderer_interface))                                      \
    PTS_PLUGIN_DEFINE(PluginClass, PTS_PLUGIN_KIND_RENDERER, Id, Name, Version)