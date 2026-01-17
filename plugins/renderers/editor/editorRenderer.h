#pragma once

#include <core/rendering/pluginUtils.hpp>

class EditorRendererPlugin final : public pts::IPlugin, public pts::IRenderer {
   public:
    bool on_load() override;
    void on_unload() override;

    void build_graph(const PtsHostApi* host, const PtsFrameParams* frame, const PtsViewParams* view,
                     const PtsFrameIO* io) override;
    void on_resize(uint32_t w, uint32_t h) override;
    void get_debug_outputs(PtsSpan* list_blob) override;
    void set_settings_blob(PtsSpan blob) override;
    void get_settings_schema(PtsSpan* schema_blob) override;
};