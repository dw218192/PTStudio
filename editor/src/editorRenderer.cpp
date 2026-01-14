#include "editorRenderer.h"

#include <core/imgui/imhelper.h>
#include <core/legacy/application.h>
#include <core/legacy/embeddedRes.h>
#include <core/legacy/enumIter.h>

#include <algorithm>

#include "editorShader.h"
#include "glslHelper.h"
#include "imgui/editorFields.h"

using namespace PTS;
using namespace PTS::Editor;

constexpr auto k_grid_dim = 100.0f;
constexpr auto k_grid_spacing = 1.0f;
GLM_CONSTEXPR auto k_clear_color = glm::vec3{0};
constexpr auto k_outline_scale = 1.02f;
GLM_CONSTEXPR auto k_outline_color = glm::vec3{1, 0, 0};
constexpr auto k_sprite_scale = 0.35f;

constexpr float k_quad_data_pos_uv[] = {
    // First triangle (positions)
    -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
    // Second triangle (positions)
    -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f,
    // First triangle (UVs)
    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    // Second triangle (UVs)
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

EditorRenderer::EditorRenderer(RenderConfig config) noexcept
    : Renderer{std::move(config), "EditorRenderer"} {
    // initialize shader editing
    auto lang = TextEditor::LanguageDefinition::GLSL();
    for (auto const k : k_glsl_keywords) lang.mKeywords.insert(k);
    for (auto const k : k_glsl_identifiers) {
        TextEditor::Identifier id;
        id.mDeclaration = "Built-in function";
        lang.mIdentifiers.insert(std::make_pair(k, id));
    }
    for (auto& data : m_shader_editor_data) {
        data.editor.SetLanguageDefinition(lang);
    }

    m_fixed_glsl_src_editor_data.emplace(k_light_inc_path, PerTextEditorData{});
    m_fixed_glsl_src_editor_data.emplace(k_uniform_inc_path, PerTextEditorData{});

    for (auto& [src_name, src_editor_data] : m_fixed_glsl_src_editor_data) {
        auto& editor = src_editor_data.editor;
        editor.SetLanguageDefinition(lang);
        if (auto src_res = try_get_embedded_res(cmrc::editor_resources::get_filesystem(),
                                                std::string{src_name})) {
            auto src = GLSLHelper::preprocess(src_res.value());
            editor.SetText(src);
        } else {
            editor.SetText(src_res.error());
        }
        editor.SetReadOnly(true);
    }

    // callbacks
    Light::get_field_info<Light::FieldTag::INTENSITY>().get_on_change_callback_list() +=
        m_on_light_intensity_change;

    Light::get_field_info<Light::FieldTag::COLOR>().get_on_change_callback_list() +=
        m_on_light_color_change;

    SceneObject::get_field_info<SceneObject::FieldTag::WORLD_TRANSFORM>()
        .get_on_change_callback_list() += m_on_obj_world_trans_change;

    SceneObject::get_field_info<SceneObject::FieldTag::LOCAL_TRANSFORM>()
        .get_on_change_callback_list() += m_on_obj_local_trans_change;

    auto upload_light_data = [this](GLBufferRef& buf, Light const*, LightData const&) {
        TL_CHECK_NON_FATAL(m_app, pts::LogLevel::Error, buf->bind());
        TL_CHECK_NON_FATAL(
            m_app, pts::LogLevel::Error,
            buf->set_data(tcb::span{m_light_data_link.data(), m_light_data_link.size()},
                          GL_DYNAMIC_DRAW));
        buf->unbind();
    };

    m_light_data_link.get_on_push_back_callbacks() += upload_light_data;
    m_light_data_link.get_on_erase_callbacks() += upload_light_data;
    m_light_data_link.get_on_update_callbacks() += upload_light_data;
}

EditorRenderer::~EditorRenderer() noexcept {
    Light::get_field_info<Light::FieldTag::INTENSITY>().get_on_change_callback_list() -=
        m_on_light_intensity_change;

    Light::get_field_info<Light::FieldTag::COLOR>().get_on_change_callback_list() -=
        m_on_light_color_change;

    SceneObject::get_field_info<SceneObject::FieldTag::WORLD_TRANSFORM>()
        .get_on_change_callback_list() -= m_on_obj_world_trans_change;

    SceneObject::get_field_info<SceneObject::FieldTag::LOCAL_TRANSFORM>()
        .get_on_change_callback_list() -= m_on_obj_local_trans_change;

    if (m_scene) {
        m_scene->get_callback_list(SceneChangeType::OBJECT_ADDED) -= m_on_add_obj;
        m_scene->get_callback_list(SceneChangeType::OBJECT_REMOVED) -= m_on_remove_obj;
    }
}

auto EditorRenderer::init(ObserverPtr<Application> app) noexcept
    -> tl::expected<void, std::string> {
    TL_CHECK_AND_PASS(Renderer::init(app));
    // load shader sources
    auto const embedded_fs = cmrc::editor_resources::get_filesystem();
    auto grid_vs_src = std::string{};
    auto grid_fs_src = std::string{};
    auto billboard_vs_src = std::string{};
    auto billboard_fs_src = std::string{};

    TL_TRY_ASSIGN(grid_vs_src, try_get_embedded_res(embedded_fs, k_grid_vs_path));
    TL_TRY_ASSIGN(grid_fs_src, try_get_embedded_res(embedded_fs, k_grid_fs_path));
    TL_TRY_ASSIGN(billboard_vs_src, try_get_embedded_res(embedded_fs, k_billboard_vs_path));
    TL_TRY_ASSIGN(billboard_fs_src, try_get_embedded_res(embedded_fs, k_billboard_fs_path));

    auto create_grid = [this, &grid_vs_src, &grid_fs_src](
                           float grid_dim, float spacing) -> tl::expected<void, std::string> {
        std::vector<glm::vec3> vertices;
        auto const half_dim = grid_dim / 2.0f;
        for (auto x = -half_dim; x <= half_dim; x += spacing) {
            vertices.emplace_back(x, 0.0f, -half_dim);
            vertices.emplace_back(x, 0.0f, half_dim);
        }
        for (auto z = -half_dim; z <= half_dim; z += spacing) {
            vertices.emplace_back(-half_dim, 0.0f, z);
            vertices.emplace_back(half_dim, 0.0f, z);
        }
        auto const& view = vertices;
        TL_TRY_ASSIGN(
            m_grid_render_data,
            GLVertexArray::create(tcb::make_span(view), GLAttributeInfo<glm::vec3>{0, 0, 0}));
        m_grid_render_data->unbind();

        CHECK_GL_ERROR();

        // grid shaders
        {
            ShaderProgram::ShaderDesc descs{
                {ShaderType::Vertex, grid_vs_src},
                {ShaderType::Fragment, grid_fs_src},
            };
            TL_TRY_ASSIGN(m_grid_shader, ShaderProgram::from_srcs(descs));
        }
        TL_CHECK_AND_PASS(m_grid_shader->bind());
        { TL_CHECK_AND_PASS(m_grid_shader->set_uniform(k_uniform_half_grid_dim, half_dim)); }
        m_grid_shader->unbind();

        return {};
    };

    if (valid()) {
        return {};
    }

    // frame buffer
    TL_TRY_ASSIGN(m_render_buf, GLFrameBuffer::create());
    TL_CHECK(m_render_buf->bind());
    {
        TL_CHECK(m_render_buf->attach(m_config.width, m_config.height,
                                      {{GL_COLOR_ATTACHMENT0,
                                        GL_RGBA,
                                        {
                                            {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                                            {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
                                        }},
                                       {GL_DEPTH_ATTACHMENT,
                                        GL_DEPTH_COMPONENT,
                                        {
                                            {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                                            {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
                                        }}}));
        TL_CHECK(m_render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));
    }
    m_render_buf->unbind();

    // default shaders
    auto default_shader_srcs = PTS::EArray<ShaderType, std::optional<std::string>>{};
    for (auto const type : EIter<ShaderType>{}) {
        auto src = std::string{};
        TL_TRY_ASSIGN(src, try_get_embedded_res(cmrc::editor_resources::get_filesystem(),
                                                k_user_shader_paths[type]));
        default_shader_srcs[type] = GLSLHelper::preprocess(src);
    }

    TL_TRY_ASSIGN(m_default_shader, ShaderProgram::from_srcs(default_shader_srcs));
    TL_CHECK_AND_PASS(create_grid(k_grid_dim, k_grid_spacing));

    if (!m_grid_shader->valid()) {
        return TL_ERROR("invalid grid shader");
    } else if (!m_default_shader->valid()) {
        return TL_ERROR("invalid editor shader");
    }

    // set up light ubo
    TL_TRY_ASSIGN(m_light_data_link.get_user_data(), GLBuffer::create(GL_UNIFORM_BUFFER));

    // set up gizmo sprite data
    if (!embedded_fs.exists(k_light_icon_png_path)) {
        return TL_ERROR("builtin icon {} not found", k_light_icon_png_path);
    }

    auto light_icon_png_data = std::string{};
    TL_TRY_ASSIGN(light_icon_png_data, try_get_embedded_res(embedded_fs, k_light_icon_png_path));
    TL_TRY_ASSIGN(
        m_light_gizmo_data.texture,
        GLTexture::create(tcb::span{light_icon_png_data.data(),
                                    light_icon_png_data.data() + light_icon_png_data.size()},
                          FileFormat::PNG));
    TL_TRY_ASSIGN(m_light_gizmo_data.render_data,
                  GLVertexArray::create(tcb::make_span(k_quad_data_pos_uv),
                                        GLAttributeInfo<glm::vec3>{0, 0, 0},
                                        GLAttributeInfo<glm::vec2>{1, 0, 18 * sizeof(float)}));
    {
        ShaderProgram::ShaderDesc descs{
            {ShaderType::Vertex, billboard_vs_src},
            {ShaderType::Fragment, billboard_fs_src},
        };
        TL_TRY_ASSIGN(m_light_gizmo_data.shader, ShaderProgram::from_srcs(descs));
    }

    // set up fonts for text editor
    // ImGui will use Font[0] as the default font
    ImGui::GetIO().Fonts->AddFontDefault();
    for (auto sz = SharedTextEditorData::min_font_size; sz <= SharedTextEditorData::max_font_size;
         ++sz) {
        auto& font = SharedTextEditorData::fonts[sz - SharedTextEditorData::min_font_size];
        font = ImGui::GetIO().Fonts->AddFontFromMemoryCompressedBase85TTF(
            consolas_compressed_data_base85, static_cast<float>(sz));
    }
    for (auto sz = SharedTextEditorData::min_font_size; sz <= SharedTextEditorData::max_font_size;
         ++sz) {
        auto& str = SharedTextEditorData::font_size_strs[sz - SharedTextEditorData::min_font_size];
        str = std::to_string(sz) + "px";
    }

    m_valid = true;
    return {};
}

auto EditorRenderer::open_scene(Ref<Scene> scene) noexcept -> tl::expected<void, std::string> {
    m_cur_outline_obj = nullptr;

    if (m_scene) {
        m_scene->get_callback_list(SceneChangeType::OBJECT_ADDED) -= m_on_add_obj;
        m_scene->get_callback_list(SceneChangeType::OBJECT_REMOVED) -= m_on_remove_obj;
    }
    m_scene = &scene.get();
    m_scene->get_callback_list(SceneChangeType::OBJECT_ADDED) += m_on_add_obj;
    m_scene->get_callback_list(SceneChangeType::OBJECT_REMOVED) += m_on_remove_obj;

    clear_render_data();

    CHECK_GL_ERROR();

    for (auto&& editable : scene.get().get_editables()) {
        on_add_obj(editable);
    }

    // Set a few settings/modes in OpenGL rendering
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POLYGON_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    CHECK_GL_ERROR();

    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    CHECK_GL_ERROR();

    return {};
}

auto EditorRenderer::on_change_render_config() noexcept -> tl::expected<void, std::string> {
    if (m_render_buf) {
        TL_CHECK_AND_PASS(m_render_buf->bind());
        { TL_CHECK_AND_PASS(m_render_buf->resize(m_config.width, m_config.height)); }
        m_render_buf->unbind();
    }
    if (m_outline.render_buf) {
        TL_CHECK_AND_PASS(m_outline.render_buf->bind());
        { TL_CHECK_AND_PASS(m_outline.render_buf->resize(m_config.width, m_config.height)); }
        m_outline.render_buf->unbind();
    }
    return {};
}

#pragma region Scene_Change_Callbacks

auto EditorRenderer::on_add_obj(Ref<SceneObject> obj) noexcept -> void {
    if (auto const render_obj = obj.get().as<RenderableObject>()) {
        if (m_obj_data.count(render_obj)) {
            m_app->log(pts::LogLevel::Error, "object added twice");
            return;
        }
        TL_CHECK_NON_FATAL(m_app, pts::LogLevel::Error,
                           on_add_object_internal(m_obj_data[render_obj], *render_obj));
    } else if (auto const light = obj.get().as<Light>()) {
        TL_CHECK_NON_FATAL(m_app, pts::LogLevel::Error,
                           m_light_data_link.push_back(light, light->get_data()));
    }
}

auto EditorRenderer::on_remove_obj(Ref<SceneObject> obj) noexcept -> void {
    if (auto const render_obj = obj.get().as<RenderableObject>()) {
        auto const it = m_obj_data.find(render_obj);
        if (it == m_obj_data.end()) {
            m_app->log(pts::LogLevel::Error, "object not found");
            return;
        }
        m_obj_data.erase(it);
        if (render_obj == m_cur_outline_obj) {
            m_cur_outline_obj = nullptr;
        }
    } else if (auto const light = obj.get().as<Light>()) {
        TL_CHECK_NON_FATAL(m_app, pts::LogLevel::Error, m_light_data_link.erase(light));
    }
}

auto EditorRenderer::update_light(Light const& light) -> void {
    auto idx = size_t{};
    TL_TRY_ASSIGN_NON_FATAL(idx, m_app, pts::LogLevel::Error, m_light_data_link.get_idx(&light));
    try {
        m_light_data_link[idx] = light.get_data();
    } catch (std::out_of_range const& err) {
        m_app->log(pts::LogLevel::Error, err.what());
    }
}

auto EditorRenderer::on_light_color_change(Light::callback_data_t<Light::FieldTag::COLOR> data)
    -> void {
    update_light(data.obj);
}

auto EditorRenderer::on_light_intensity_change(
    Light::callback_data_t<Light::FieldTag::INTENSITY> data) -> void {
    update_light(data.obj);
}

auto EditorRenderer::on_obj_world_trans_change(
    SceneObject::callback_data_t<SceneObject::FieldTag::WORLD_TRANSFORM> data) -> void {
    if (auto const light = data.obj.as<Light>()) {
        update_light(*light);
    }
}

auto EditorRenderer::on_obj_local_trans_change(
    SceneObject::callback_data_t<SceneObject::FieldTag::LOCAL_TRANSFORM> data) -> void {
    if (auto const light = data.obj.as<Light>()) {
        update_light(*light);
    }
}

void EditorRenderer::on_selected_editable_change(ObserverPtr<SceneObject> editable) noexcept {
    commit_cur_shader_code();
    m_cur_outline_obj = nullptr;

    if (!editable) {
    } else if (auto const obj = editable->as<RenderableObject>()) {
        auto const it = m_obj_data.find(obj);
        if (it == m_obj_data.end()) {
            return;
        }
        // save last editing to the object editing data
        m_cur_outline_obj = obj;

        // update editor texts
        for (auto const shader_type : EIter<ShaderType>{}) {
            m_shader_editor_data[shader_type].editor.SetText(
                it->second.editing_data.get_src(shader_type));
        }
    }
}

auto EditorRenderer::on_add_object_internal(
    PerObjectData& data, RenderableObject const& obj) noexcept -> tl::expected<void, std::string> {
    TL_TRY_ASSIGN(data.shader, ShaderProgram::clone(m_default_shader.get()));
    TL_TRY_ASSIGN(data.render_data,
                  GLVertexArray::create_indexed(
                      obj.get_vertices(), obj.get_indices(),
                      GLAttributeInfo<glm::vec3>{EditorShader::VertexAttribBinding::Position,
                                                 sizeof(Vertex), offsetof(Vertex, position)},
                      GLAttributeInfo<glm::vec3>{EditorShader::VertexAttribBinding::Normal,
                                                 sizeof(Vertex), offsetof(Vertex, normal)},
                      GLAttributeInfo<glm::vec3>{EditorShader::VertexAttribBinding::TexCoords,
                                                 sizeof(Vertex), offsetof(Vertex, uv)}));

    // update editor texts
    for (auto const shader_type : EIter<ShaderType>{}) {
        m_shader_editor_data[shader_type].editor.SetText(data.editing_data.get_src(shader_type));
    }

    return {};
}

#pragma endregion Scene_Change_Callbacks

auto EditorRenderer::draw_outline(View<Camera> cam_view,
                                  View<RenderableObject> obj) -> tl::expected<void, std::string> {
    auto& cam = cam_view.get();
    if (!m_outline.render_buf) {
        // initialize outline states

        // render buffer
        TL_TRY_ASSIGN(m_outline.render_buf, GLFrameBuffer::create());
        TL_CHECK_AND_PASS(m_outline.render_buf->bind());
        {
            TL_CHECK_AND_PASS(
                m_outline.render_buf->attach(m_config.width, m_config.height,
                                             {{GL_COLOR_ATTACHMENT0,
                                               GL_RGBA,
                                               {
                                                   {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                                                   {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
                                                   {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
                                                   {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
                                               }},
                                              {GL_COLOR_ATTACHMENT1,
                                               GL_RGBA,
                                               {
                                                   {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                                                   {GL_TEXTURE_MAG_FILTER, GL_LINEAR},
                                                   {GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE},
                                                   {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE},
                                               }}}));
            TL_CHECK_AND_PASS(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));
        }
        m_outline.render_buf->unbind();

        // shaders
        auto const embedded_fs = cmrc::editor_resources::get_filesystem();
        for (size_t i = 0; i < m_outline.shaders.size(); ++i) {
            if (!embedded_fs.exists(k_outline_vs_paths[i])) {
                return TL_ERROR("built-in shader {} not found", k_outline_vs_paths[i]);
            }
            if (!embedded_fs.exists(k_outline_fs_paths[i])) {
                return TL_ERROR("built-in shader {} not found", k_outline_fs_paths[i]);
            }

            auto const vs_src = embedded_fs.open(k_outline_vs_paths[i]);
            auto const fs_src = embedded_fs.open(k_outline_fs_paths[i]);

            ShaderProgram::ShaderDesc descs{
                {ShaderType::Vertex, std::string{vs_src.begin(), vs_src.size()}},
                {ShaderType::Fragment, std::string{fs_src.begin(), fs_src.size()}}};
            TL_TRY_ASSIGN(m_outline.shaders[i], ShaderProgram::from_srcs(descs));
        }

        // full screen quad
        TL_TRY_ASSIGN(m_outline.quad_render_data,
                      GLVertexArray::create(tcb::make_span(k_quad_data_pos_uv),
                                            GLAttributeInfo<glm::vec3>{0, 0, 0},
                                            GLAttributeInfo<glm::vec2>{1, 0, 18 * sizeof(float)}));
        m_outline.quad_render_data->unbind();
    }

    TL_CHECK_AND_PASS(m_outline.render_buf->bind());
    {
        TL_CHECK_AND_PASS(m_outline.render_buf->clear(glm::vec3(0), 1.0f));

        // draw to attachment 0
        TL_CHECK(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));

        TL_CHECK_AND_PASS(m_outline.shaders[0]->bind());
        {
            TL_CHECK_AND_PASS(m_outline.shaders[0]->set_uniform(k_uniform_view, cam.get_view()));
            TL_CHECK_AND_PASS(
                m_outline.shaders[0]->set_uniform(k_uniform_projection, cam.get_projection()));
            TL_CHECK_AND_PASS(m_outline.shaders[0]->set_uniform(
                k_uniform_model, obj.get().get_transform(TransformSpace::WORLD).get_matrix()));

            auto res = try_get_obj_data(obj);
            if (!res) return TL_ERROR(res.error());

            auto&& data = res.value().get();
            auto&& vao = data.render_data;
            TL_CHECK_AND_PASS(vao->bind());
            TL_CHECK_AND_PASS(vao->draw(GL_TRIANGLES));
        }
        // draw to attachment 1 based on attachment 0
        TL_CHECK(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT1));
        TL_CHECK_AND_PASS(m_outline.shaders[1]->bind());
        {
            TL_CHECK_AND_PASS(m_outline.shaders[1]->set_texture(
                k_uniform_screen_texture, m_outline.render_buf->get_texture(GL_COLOR_ATTACHMENT0),
                0));
            TL_CHECK_AND_PASS(
                m_outline.shaders[1]->set_uniform(k_uniform_outline_color, glm::vec3(1, 0, 0)));
            TL_CHECK_AND_PASS(m_outline.shaders[1]->set_uniform(k_uniform_thickness, 0.9f));
            TL_CHECK_AND_PASS(m_outline.shaders[1]->set_uniform(
                k_uniform_texel_size, 1.0f / glm::vec2(m_config.width, m_config.height)));
            TL_CHECK_AND_PASS(m_outline.quad_render_data->bind());
            TL_CHECK_AND_PASS(m_outline.quad_render_data->draw(GL_TRIANGLES));
            m_outline.quad_render_data->unbind();
        }
    }
    m_outline.render_buf->unbind();

    // attachment 1 now contains the outline
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    TL_CHECK_AND_PASS(m_render_buf->bind());
    TL_CHECK_AND_PASS(m_outline.shaders[2]->bind());
    {
        TL_CHECK_AND_PASS(m_outline.quad_render_data->bind());
        {
            TL_CHECK_AND_PASS(m_outline.shaders[2]->set_texture(
                k_uniform_screen_texture, m_outline.render_buf->get_texture(GL_COLOR_ATTACHMENT1),
                0));
            TL_CHECK_AND_PASS(m_outline.quad_render_data->draw(GL_TRIANGLES));
        }
        m_outline.quad_render_data->unbind();
    }
    m_outline.shaders[2]->unbind();

    glDisable(GL_BLEND);

    return {};
}

auto EditorRenderer::render(View<Camera> cam_view) noexcept
    -> tl::expected<TextureHandle, std::string> {
    auto& cam = cam_view.get();

    if (!valid()) {
        return TL_ERROR("invalid EditorRenderer");
    }

    TL_CHECK(m_render_buf->bind());
    TL_CHECK(m_render_buf->clear(k_clear_color, 1.0f));

    glViewport(0, 0, static_cast<GLsizei>(get_config().width),
               static_cast<GLsizei>(get_config().height));
    CHECK_GL_ERROR();

    // render objects
    for (auto const& obj : m_scene->get_objects_of_type<RenderableObject>()) {
        auto res = try_get_obj_data(obj);
        if (!res) {
            return TL_ERROR(res.error());
        }

        auto& data = res.value().get();
        auto& shader = data.shader;
        auto& vao = data.render_data;

        TL_CHECK(shader->bind());

        auto const& uniforms = shader->get_uniform_map().get();
        // light
        if (uniforms.count(k_uniform_light_count)) {
            TL_CHECK(shader->set_uniform(k_uniform_light_count,
                                         static_cast<int>(m_light_data_link.size())));

            glBindBufferBase(GL_UNIFORM_BUFFER, EditorShader::UBOBinding::LightBlock,
                             m_light_data_link.get_user_data()->handle());
            CHECK_GL_ERROR();
        }
        // material
        if (uniforms.count(k_uniform_object_color)) {
            TL_CHECK(shader->set_uniform(k_uniform_object_color, obj.get_material().albedo));
        }
        // MVP
        if (uniforms.count(k_uniform_view)) {
            TL_CHECK(shader->set_uniform(k_uniform_view, cam.get_view()));
        }
        if (uniforms.count(k_uniform_projection)) {
            TL_CHECK(shader->set_uniform(k_uniform_projection, cam.get_projection()));
        }
        if (uniforms.count(k_uniform_model)) {
            TL_CHECK(shader->set_uniform(k_uniform_model,
                                         obj.get_transform(TransformSpace::WORLD).get_matrix()));
        }
        // misc
        if (uniforms.count(k_uniform_time)) {
            TL_CHECK(shader->set_uniform(k_uniform_time, m_app->get_time()));
        }
        if (uniforms.count(k_uniform_delta_time)) {
            TL_CHECK(shader->set_uniform(k_uniform_delta_time, m_app->get_delta_time()));
        }
        if (uniforms.count(k_uniform_resolution)) {
            TL_CHECK(shader->set_uniform(k_uniform_resolution,
                                         glm::ivec2{get_config().width, get_config().height}));
        }

        TL_CHECK(vao->bind());
        TL_CHECK(vao->draw(GL_TRIANGLES));
    }

    // render grid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    TL_CHECK(m_grid_render_data->bind());
    TL_CHECK(m_grid_shader->bind());
    {
        TL_CHECK(m_grid_shader->set_uniform(k_uniform_view, cam.get_view()));
        TL_CHECK(m_grid_shader->set_uniform(k_uniform_projection, cam.get_projection()));
        TL_CHECK(m_grid_render_data->draw(GL_LINES));
    }

    // render gizmo
    if (!m_scene->get_objects_of_type<Light>().empty()) {
        glDisable(GL_DEPTH_TEST);
        TL_CHECK(m_light_gizmo_data.render_data->bind());
        TL_CHECK(m_light_gizmo_data.shader->bind());
        {
            TL_CHECK(m_light_gizmo_data.shader->set_uniform(k_uniform_view, cam.get_view()));
            TL_CHECK(
                m_light_gizmo_data.shader->set_uniform(k_uniform_projection, cam.get_projection()));
            TL_CHECK(m_light_gizmo_data.shader->set_texture(k_uniform_sprite_texture,
                                                            m_light_gizmo_data.texture.get(), 0));
            TL_CHECK(
                m_light_gizmo_data.shader->set_uniform(k_uniform_sprite_scale, k_sprite_scale));

            // draw light gizmos
            for (auto&& light : m_scene->get_objects_of_type<Light>()) {
                TL_CHECK(m_light_gizmo_data.shader->set_uniform(
                    k_uniform_sprite_world_pos,
                    light.get_transform(TransformSpace::WORLD).get_position()));
                TL_CHECK(m_light_gizmo_data.shader->set_uniform(k_uniform_sprite_tint,
                                                                light.get_color()));
                TL_CHECK(m_light_gizmo_data.render_data->draw(GL_TRIANGLES));
            }
        }
        glEnable(GL_DEPTH_TEST);
    }

    // draw_outline will unbind the shader
    glDisable(GL_BLEND);

    // render outline
    if (m_cur_outline_obj) {
        TL_CHECK(draw_outline(cam_view, *m_cur_outline_obj));
    }

    m_render_buf->unbind();
    return m_render_buf->get_texture(GL_COLOR_ATTACHMENT0);
}

void EditorRenderer::clear_render_data() {
    m_obj_data.clear();
}

#pragma region Shader_Editing

void EditorRenderer::commit_cur_shader_code() noexcept {
    if (m_cur_outline_obj) {
        for (auto const shader_type : EIter<ShaderType>{}) {
            m_obj_data.at(m_cur_outline_obj)
                .editing_data.set_src(shader_type,
                                      m_shader_editor_data[shader_type].editor.GetText());
        }
    }
}

auto EditorRenderer::try_compile() noexcept -> void {
    auto descs = ShaderProgram::ShaderDesc{};

    auto& data = m_obj_data.at(m_cur_outline_obj);
    auto& edit_data = data.editing_data;

    for (auto const shader_type : EIter<ShaderType>{}) {
        auto main_src = m_shader_editor_data[shader_type].editor.GetText();
        // if it has text, we'll try to compile it
        if (!main_src.empty()) {
            descs[shader_type] = GLSLHelper::preprocess(main_src);
        }
    }

    auto res = data.shader->try_recompile(descs);
    if (!res) {
        m_app->log(pts::LogLevel::Error, res.error());
        edit_data.compilation_status = PerObjectEditingData::CompilationStatus::FAILURE;
    } else {
        commit_cur_shader_code();
        edit_data.compilation_status = PerObjectEditingData::CompilationStatus::SUCCESS;
    }
}

auto EditorRenderer::draw_text_editor_header(Ref<PerTextEditorData> editor_ref) noexcept -> void {
    auto&& editor_data = editor_ref.get();
    auto&& editor = editor_data.editor;

    if (ImGui::Button("Compile")) {
        try_compile();
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::CalcTextSize("00px").x * 3.0f +
                            ImGui::GetStyle().FramePadding.x * 2.0f);

    if (ImGui::BeginCombo("Font Size", SharedTextEditorData::get_font_size_str(
                                           m_shared_editor_data.cur_font_size))) {
        for (int sz = SharedTextEditorData::min_font_size;
             sz <= SharedTextEditorData::max_font_size; ++sz) {
            auto const selected = m_shared_editor_data.cur_font_size == sz;
            if (ImGui::Selectable(SharedTextEditorData::get_font_size_str(sz), selected)) {
                m_shared_editor_data.cur_font_size = sz;
            }
        }
        ImGui::EndCombo();
    }

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save")) {
                auto textToSave = editor.GetText();
                // TODO: save
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            bool ro = editor.IsReadOnly();
            if (ImGui::MenuItem("Read-only mode", nullptr, &ro)) {
                editor.SetReadOnly(ro);
            }
            ImGui::Separator();

            if (ImGui::MenuItem("Undo", "ALT-Backspace", nullptr, !ro && editor.CanUndo())) {
                editor.Undo();
            }
            if (ImGui::MenuItem("Redo", "Ctrl-Y", nullptr, !ro && editor.CanRedo())) {
                editor.Redo();
            }
            ImGui::Separator();

            if (ImGui::MenuItem("Copy", "Ctrl-C", nullptr, editor.HasSelection())) {
                editor.Copy();
            }
            if (ImGui::MenuItem("Cut", "Ctrl-X", nullptr, !ro && editor.HasSelection())) {
                editor.Cut();
            }
            if (ImGui::MenuItem("Delete", "Del", nullptr, !ro && editor.HasSelection())) {
                editor.Delete();
            }
            if (ImGui::MenuItem("Paste", "Ctrl-V", nullptr,
                                !ro && ImGui::GetClipboardText() != nullptr)) {
                editor.Paste();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Select all", nullptr, nullptr))
                editor.SetSelection(TextEditor::Coordinates(),
                                    TextEditor::Coordinates(editor.GetTotalLines(), 0));

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Dark palette")) editor.SetPalette(TextEditor::GetDarkPalette());
            if (ImGui::MenuItem("Light palette")) editor.SetPalette(TextEditor::GetLightPalette());
            if (ImGui::MenuItem("Retro blue palette"))
                editor.SetPalette(TextEditor::GetRetroBluePalette());
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}

auto EditorRenderer::draw_text_editor(Ref<PerTextEditorData> editor_ref) const noexcept -> void {
    auto&& editor_data = editor_ref.get();
    auto& editor = editor_data.editor;

    auto const cpos = editor.GetCursorPosition();
    ImGui::SameLine();
    ImGui::Text("%6d/%-6d %6d lines  | %s | %s | %s | %s", cpos.mLine + 1, cpos.mColumn + 1,
                editor.GetTotalLines(), editor.IsOverwrite() ? "Ovr" : "Ins",
                editor.CanUndo() ? "*" : " ", editor.GetLanguageDefinition().mName.c_str(),
                m_cur_outline_obj->get_name().data());

    ImGui::PushFont(SharedTextEditorData::get_font(m_shared_editor_data.cur_font_size));
    editor.Render("GLSL Editor");
    ImGui::PopFont();
}

auto EditorRenderer::draw_glsl_editor(ShaderType type, Ref<ShaderProgram> shader_ref,
                                      Ref<PerTextEditorData> editor_ref) noexcept
    -> tl::expected<void, std::string> {
    auto& obj_data = m_obj_data[m_cur_outline_obj];
    auto& shader = shader_ref.get();

    draw_text_editor_header(editor_ref);

    auto const uniforms = shader.get_uniform_map();
    ImGui::SetNextItemOpen(m_shared_editor_data.show_uniform);
    if ((m_shared_editor_data.show_uniform = ImGui::CollapsingHeader("Uniform Variables"))) {
        if (ImGui::RadioButton("Show Built-in", m_shared_editor_data.show_built_in_uniform)) {
            m_shared_editor_data.show_built_in_uniform =
                !m_shared_editor_data.show_built_in_uniform;
        }

        if (ImGui::BeginChild("##Uniform Variables", ImVec2(0, 150), true)) {
            for (auto const& [name, val] : uniforms.get()) {
                auto disabled =
                    std::find(std::begin(k_built_in_uniforms), std::end(k_built_in_uniforms),
                              name) != std::end(k_built_in_uniforms);

                if (disabled && !m_shared_editor_data.show_built_in_uniform) {
                    continue;
                }
                auto val_copy = val;
                ImGui::BeginDisabled(disabled);
                if (disabled) {
                    auto display_name = name + " (built-in)";
                    ImGui::ShaderVariableField(display_name.c_str(), val_copy);
                } else {
                    if (ImGui::ShaderVariableField(name.c_str(), val_copy)) {
                        TL_CHECK(shader.bind());
                        TL_CHECK_NON_FATAL(m_app, pts::LogLevel::Warning,
                                           shader.set_uniform(name, val_copy));
                        shader.unbind();
                    }
                }
                ImGui::EndDisabled();
                ImGui::Separator();
                ImGui::Spacing();
            }
        }
        ImGui::EndChild();
    }

    ImGui::Separator();
    draw_text_editor(editor_ref);

    return {};
}

auto EditorRenderer::try_get_obj_data(View<RenderableObject> obj) noexcept
    -> tl::expected<Ref<PerObjectData>, std::string> {
    auto const it = m_obj_data.find(&obj.get());
    if (it == m_obj_data.end()) {
        return TL_ERROR("obj not found in render data");
    }
    return it->second;
}

auto EditorRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
    TL_CHECK_AND_PASS(Renderer::draw_imgui());

    if (ImGui::Begin("Shader Editor", nullptr,
                     ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_MenuBar)) {
        ImGui::SetWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
        ImGui::SetWindowPos(ImVec2(700, 300), ImGuiCond_FirstUseEver);

        if (!m_cur_outline_obj) {
            ImGui::Text("Please Select an object first");
        } else {
            switch (m_obj_data[m_cur_outline_obj].editing_data.compilation_status) {
                case PerObjectEditingData::CompilationStatus::FAILURE:
                    ImGui::GetForegroundDrawList()->AddRect(
                        ImGui::GetWindowContentRegionMin() + ImGui::GetWindowPos(),
                        ImGui::GetWindowContentRegionMax() + ImGui::GetWindowPos(),
                        IM_COL32(255, 0, 0, 255));
                    break;
                case PerObjectEditingData::CompilationStatus::SUCCESS:
                    ImGui::GetForegroundDrawList()->AddRect(
                        ImGui::GetWindowContentRegionMin() + ImGui::GetWindowPos(),
                        ImGui::GetWindowContentRegionMax() + ImGui::GetWindowPos(),
                        IM_COL32(0, 255, 0, 255));
                    break;
                case PerObjectEditingData::CompilationStatus::UNKNOWN:
                    break;
            }

            // draw a GLSL editor for each shader type
            if (ImGui::BeginTabBar("##tab")) {
                for (auto const shader_type : EIter<ShaderType>{}) {
                    if (ImGui::BeginTabItem(to_string(shader_type))) {
                        auto res = try_get_obj_data(*m_cur_outline_obj);
                        if (!res) return TL_ERROR(res.error());

                        draw_glsl_editor(shader_type, *(res.value().get().shader),
                                         m_shader_editor_data[shader_type]);
                        ImGui::EndTabItem();
                    }
                }
                // draw a GLSL editor for common GLSL funcs
                for (auto& [src_name, src_editor_data] : m_fixed_glsl_src_editor_data) {
                    if (ImGui::BeginTabItem(
                            src_name.substr(src_name.find_last_of('/') + 1).data())) {
                        draw_text_editor_header(src_editor_data);
                        draw_text_editor(src_editor_data);
                        ImGui::EndTabItem();
                    }
                }
                ImGui::EndTabBar();
            }
        }
    }

    ImGui::End();

    return {};
}

PerObjectEditingData::PerObjectEditingData() {
    for (auto const type : EIter<ShaderType>{}) {
        if (auto const src_res = try_get_embedded_res(cmrc::editor_resources::get_filesystem(),
                                                      k_user_shader_paths[type])) {
            set_src(type, std::move(src_res.value()));
        } else {
            set_src(type, src_res.error());
        }
    }
}

void PerObjectEditingData::set_src(ShaderType type, std::string src) {
    m_shader_srcs[type] = std::move(src);
}

auto PerObjectEditingData::get_src(ShaderType type) -> View<std::string> {
    return m_shader_srcs[type];
}
#pragma endregion Shader_Editing
