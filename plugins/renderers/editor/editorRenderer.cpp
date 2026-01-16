#include "editorRenderer.h"

#include <core/rendering/graph.h>
#include <core/rendering/types.h>

namespace {
void encode_noop(PtsCmd*, void*) {
}
}  // namespace

bool EditorRendererPlugin::on_load() {
    if (logger().is_valid()) {
        logger().log_info("Editor renderer plugin loaded");
    }
    return true;
}

void EditorRendererPlugin::on_unload() {
    if (logger().is_valid()) {
        logger().log_info("Editor renderer plugin unloaded");
    }
}

void EditorRendererPlugin::build_graph(const PtsHostApi* host, const PtsFrameParams*,
                                       const PtsViewParams*, const PtsFrameIO* io) {
    if (!host || !io || PTS_IS_NULL(io->output)) {
        return;
    }

    auto* graph_api = static_cast<const PtsRenderGraphApi*>(host->render_graph_api);
    if (!graph_api) {
        return;
    }

    auto g = graph_api->begin();
    if (PTS_IS_NULL(g)) {
        return;
    }

    auto output = graph_api->import_texture(g, io->output, "editor.output");
    if (!PTS_IS_NULL(output)) {
        PtsTextureViewDesc view_desc{};
        view_desc.tex = output;
        view_desc.view_format = PTS_FMT_UNKNOWN;
        view_desc.view_type = PTS_VIEW_RTV;
        view_desc.range = {0, 1, 0, 1};

        auto view = graph_api->create_tex_view(g, &view_desc, "editor.output.view");
        if (!PTS_IS_NULL(view)) {
            PtsResUse res_use{};
            res_use.kind = PTS_RES_TEXTURE_VIEW;
            res_use.access = PTS_ACCESS_RTV;
            res_use.tex_view = view;
            PtsAttachment color{};
            color.view = view;
            color.load_op = PTS_LOAD_CLEAR;
            color.store_op = PTS_STORE_STORE;
            color.clear.rgba[0] = 0.06f;
            color.clear.rgba[1] = 0.07f;
            color.clear.rgba[2] = 0.08f;
            color.clear.rgba[3] = 1.0f;

            PtsPassDesc pass{};
            pass.name = "editor.clear";
            pass.resources = &res_use;
            pass.resource_count = 1;
            pass.color_attachments = &color;
            pass.color_count = 1;
            pass.depth_attachment = nullptr;
            pass.encode = &encode_noop;
            pass.user = nullptr;
            graph_api->add_pass(g, &pass);
        }
    }

    graph_api->end(g);
}

void EditorRendererPlugin::on_resize(uint32_t, uint32_t) {
}

void EditorRendererPlugin::get_debug_outputs(PtsSpan* list_blob) {
    if (!list_blob) {
        return;
    }
    list_blob->data = nullptr;
    list_blob->size_bytes = 0;
}

void EditorRendererPlugin::set_settings_blob(PtsSpan) {
}

void EditorRendererPlugin::get_settings_schema(PtsSpan* schema_blob) {
    if (!schema_blob) {
        return;
    }
    schema_blob->data = nullptr;
    schema_blob->size_bytes = 0;
}
