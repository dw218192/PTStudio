#include "include/editorRenderer.h"
#include "application.h"

#include <cassert>
#include <algorithm>

#include "enumIter.h"
#include "include/imgui/editorFields.h"

constexpr auto k_grid_dim = 100.0f;
constexpr auto k_grid_spacing = 1.0f;
constexpr auto k_clear_color = glm::vec3{ 0 };
constexpr auto k_outline_scale = 1.02f;
constexpr auto k_outline_color = glm::vec3{ 1, 0, 0 };

EditorRenderer::EditorRenderer(RenderConfig config) noexcept : Renderer{std::move(config)} {

    auto lang = TextEditor::LanguageDefinition::GLSL();
    for (auto const k : glsl_keywords)
        lang.mKeywords.insert(k);
    for (auto const k : glsl_identifiers) {
        TextEditor::Identifier id;
        id.mDeclaration = "Built-in function";
        lang.mIdentifiers.insert(std::make_pair(k, id));
    }

    for(auto&& data : m_shader_editor_data) {
        data.editor.SetLanguageDefinition(lang);
    }
}

EditorRenderer::~EditorRenderer() noexcept = default;

auto EditorRenderer::init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> {
    TL_CHECK_AND_PASS(Renderer::init(app));

    auto create_grid = [this](float grid_dim, float spacing) -> tl::expected<void, std::string> {
        std::vector<glm::vec3> vertices;
        float const half_dim = grid_dim / 2.0f;
        for (float x = -half_dim; x <= half_dim; x += spacing) {
            vertices.emplace_back(x, 0.0f, -half_dim);
            vertices.emplace_back(x, 0.0f, half_dim);
        }
        for (float z = -half_dim; z <= half_dim; z += spacing) {
            vertices.emplace_back(-half_dim, 0.0f, z);
            vertices.emplace_back(half_dim, 0.0f, z);
        }

        TL_TRY_ASSIGN(m_grid_render_data, GLVertexArray::create(vertices.size()));
        TL_CHECK_AND_PASS(m_grid_render_data->bind());
    	{
            auto const& view = vertices;
            TL_CHECK_AND_PASS(m_grid_render_data->connect(tcb::make_span(view), GLAttributeInfo<glm::vec3>{0, 0, 0}));
        }
        m_grid_render_data->unbind();

        CHECK_GL_ERROR();

        // grid shaders
        {
            std::array<ShaderProgram::StageDesc<std::string_view>, 2> descs{
                {
                    {ShaderType::Vertex, vs_grid_src},
                    {ShaderType::Fragment, ps_grid_src},
                }
            };
            TL_TRY_ASSIGN(m_grid_shader, ShaderProgram::from_srcs(descs));
        }
        TL_CHECK_AND_PASS(m_grid_shader->bind());
        {
            TL_CHECK_AND_PASS(m_grid_shader->set_uniform(k_uniform_half_grid_dim, half_dim));
        }
        m_grid_shader->unbind();

        return {};
    };

	if (valid()) {
        return {};
    }

    // set up main frame buffer
    MainFrameBuffer::set(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	// default shaders
    {
	    std::array<ShaderProgram::StageDesc<std::string_view>, 2> descs{
		    {
                {ShaderType::Vertex, vs_obj_src},
                {ShaderType::Fragment, ps_obj_src},
		    }
	    };
        TL_TRY_ASSIGN(m_default_shader, ShaderProgram::from_srcs(descs));
    }
    TL_CHECK_AND_PASS(create_grid(k_grid_dim, k_grid_spacing));

    if (!m_grid_shader->valid()) {
        return TL_ERROR("invalid grid shader");
    } else if (!m_default_shader->valid()) {
        return TL_ERROR("invalid editor shader");
    }
    m_valid = true;
    return {};
}

auto EditorRenderer::open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> {
    m_cur_outline_obj = nullptr;
    m_scene = &scene.get();
    clear_render_data();

    CHECK_GL_ERROR();

    for (auto const obj : scene.get()) {
        TL_CHECK_AND_PASS(on_add_object(obj));
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

auto EditorRenderer::render(View<Camera> cam) noexcept -> tl::expected<void, std::string> {
    return render_internal(cam, 0);
}

auto EditorRenderer::render_buffered(View<Camera> cam) noexcept -> tl::expected<TextureHandle, std::string> {
    if (!m_render_buf) {
        TL_TRY_ASSIGN(m_render_buf, GLFrameBuffer::create());
        TL_CHECK(m_render_buf->bind());
        {
	        TL_CHECK(m_render_buf->attach(m_config.width, m_config.height, {
		        {
					GL_COLOR_ATTACHMENT0, GL_RGB,
			        {
				        { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
				        { GL_TEXTURE_MAG_FILTER, GL_LINEAR },
			        }
		        },
		        {
		        	GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT,
		        	{
				        { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
				        { GL_TEXTURE_MAG_FILTER, GL_LINEAR },
		        	}
		        }
		    }));
            TL_CHECK(m_render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));
        }
        m_render_buf->unbind();
    }
    TL_CHECK(render_internal(cam, m_render_buf->handle()));
    return m_render_buf->get_texture(GL_COLOR_ATTACHMENT0);
}

auto EditorRenderer::on_change_render_config(RenderConfig config) noexcept -> tl::expected<void, std::string> {
    if (!config.is_valid() || config == m_config) {
        return {};
    }
    m_config = std::move(config);
    if (m_render_buf) {
        TL_CHECK_AND_PASS(m_render_buf->bind());
        {
            TL_CHECK_AND_PASS(m_render_buf->resize(m_config.width, m_config.height));
        }
        m_render_buf->unbind();
    }
    if (m_outline.render_buf) {
        TL_CHECK_AND_PASS(m_outline.render_buf->bind());
        {
            TL_CHECK_AND_PASS(m_outline.render_buf->resize(m_config.width, m_config.height));
        }
        m_outline.render_buf->unbind();
    }
    return {};
}

auto EditorRenderer::on_add_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> {
    if (!obj) {
        return {};
    } else if (m_obj_data.count(obj)) {
        return TL_ERROR("object added twice");
    }
    TL_CHECK_AND_PASS(on_add_object_internal(m_obj_data[obj], obj));

    return {};
}

auto EditorRenderer::on_remove_object(ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> {
    auto const it = m_obj_data.find(obj);
    if(it == m_obj_data.end()) {
        return TL_ERROR("on_remove_object: obj not found");
    }
    m_obj_data.erase(obj);

    return {};
}

void EditorRenderer::on_object_change(ViewPtr<Object> obj) noexcept {
    // save last editing to the object editing data
    commit_cur_shader_code();

    m_cur_outline_obj = obj;
    for (auto const shader_type : EIter<ShaderType>{}) {
        m_shader_editor_data[shader_type].editor.SetText(
            m_obj_data[obj].editing_data.shader_srcs[shader_type]
        );
    }
}

auto EditorRenderer::on_add_object_internal(Ref<PerObjectData> data, ViewPtr<Object> obj) noexcept -> tl::expected<void, std::string> {
    TL_TRY_ASSIGN(data.get().shader, ShaderProgram::clone(m_default_shader.get()));
	TL_TRY_ASSIGN(data.get().render_data, GLVertexArray::create(obj->get_vertices().size()));

    TL_CHECK_AND_PASS(data.get().render_data->bind());
    {
        TL_CHECK_AND_PASS(data.get().render_data->connect(obj->get_vertices(),
            GLAttributeInfo<glm::vec3> { 0, sizeof(Vertex), offsetof(Vertex, position) },
            GLAttributeInfo<glm::vec3> { 1, sizeof(Vertex), offsetof(Vertex, normal) },
            GLAttributeInfo<glm::vec3> { 2, sizeof(Vertex), offsetof(Vertex, uv) }
        ));
    }
    data.get().render_data->unbind();

    for (auto const shader_type : EIter<ShaderType>{}) {
        data.get().editing_data.shader_srcs[shader_type] = data.get().shader->get_stage(shader_type)->get_src();

        m_shader_editor_data[shader_type].editor.SetText(
            m_obj_data[obj].editing_data.shader_srcs[shader_type]
        );
    }

    return {};

}

auto EditorRenderer::render_internal(View<Camera> cam, GLuint fbo) noexcept -> tl::expected<void, std::string> {
    auto draw_outline = [this, cam, fbo](ViewPtr<Object> obj)-> tl::expected<void, std::string> {
        if (!m_outline.render_buf) {
            // initialize outline states

            // render buffer
            TL_TRY_ASSIGN(m_outline.render_buf, GLFrameBuffer::create());
            TL_CHECK_AND_PASS(m_outline.render_buf->bind());
            {
                TL_CHECK_AND_PASS(m_outline.render_buf->attach(m_config.width, m_config.height, {
                    {
                    	GL_COLOR_ATTACHMENT0, GL_RGBA,
                    	{
                            { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
							{ GL_TEXTURE_MAG_FILTER, GL_LINEAR },
                            { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE },
                    		{ GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE },
                    	}
                    },
                    {
                    	GL_COLOR_ATTACHMENT1, GL_RGBA,
                    	{
                            { GL_TEXTURE_MIN_FILTER, GL_LINEAR },
                            { GL_TEXTURE_MAG_FILTER, GL_LINEAR },
                            { GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE },
                            { GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE },
                        }
                    }
                }));
                TL_CHECK_AND_PASS(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));
            }
            m_outline.render_buf->unbind();

            // shaders
            for (int i = 0; i < m_outline.shaders.size(); ++i) {
	            std::array<ShaderProgram::StageDesc<std::string_view>, 2> descs{
		            {
			            {
				            ShaderType::Vertex, vs_outline_passes[i]
			            },
			            {
				            ShaderType::Fragment, ps_outline_passes[i]
			            }
		            }
	            };
                TL_TRY_ASSIGN(m_outline.shaders[i], ShaderProgram::from_srcs(descs));
            }

            // full screen quad
            TL_TRY_ASSIGN(m_outline.quad_render_data, GLVertexArray::create(6));
            TL_CHECK_AND_PASS(m_outline.quad_render_data->bind());
            {
                static constexpr float data[] = {
                    // First triangle (positions)
                    -1.0f, -1.0f, 0.0f,
                     1.0f, -1.0f, 0.0f,
                     1.0f,  1.0f, 0.0f,
                     // Second triangle (positions)
                     -1.0f, -1.0f, 0.0f,
                      1.0f,  1.0f, 0.0f,
                     -1.0f,  1.0f, 0.0f,
                     // First triangle (UVs)
                     0.0f, 0.0f,
                     1.0f, 0.0f,
                     1.0f, 1.0f,
                     // Second triangle (UVs)
                     0.0f, 0.0f,
                     1.0f, 1.0f,
                     0.0f, 1.0f
                };
                TL_CHECK_AND_PASS(m_outline.quad_render_data->connect(tcb::make_span(data), 
                    GLAttributeInfo<glm::vec3>{0, 0, 0},
                    GLAttributeInfo<glm::vec2>{1, 0, 18 * sizeof(float)}
                ));
            }
            m_outline.quad_render_data->unbind();
        }

        TL_CHECK_AND_PASS(m_outline.render_buf->bind());
        {
            TL_CHECK_AND_PASS(m_outline.render_buf->clear(glm::vec3(0), 1.0f));

            // draw to attachment 0
            TL_CHECK(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT0));

            TL_CHECK_AND_PASS(m_outline.shaders[0]->bind());
            {
                TL_CHECK_AND_PASS(m_outline.shaders[0]->set_uniform(k_uniform_view, cam.get().get_view()));
                TL_CHECK_AND_PASS(m_outline.shaders[0]->set_uniform(k_uniform_projection, cam.get().get_projection()));
                TL_CHECK_AND_PASS(m_outline.shaders[0]->set_uniform(k_uniform_model, obj->get_transform().get_matrix()));

                auto res = try_get_obj_data(obj);
                if (!res) return TL_ERROR(res.error());

                auto&& data = res.value().get();
                auto&& vao = data.render_data;
                TL_CHECK_AND_PASS(vao->bind());
                TL_CHECK_AND_PASS(vao->draw_array(GL_TRIANGLES));
            }
            // draw to attachment 1 based on attachment 0
            TL_CHECK(m_outline.render_buf->set_draw_buffer(GL_COLOR_ATTACHMENT1));
            TL_CHECK_AND_PASS(m_outline.shaders[1]->bind());
            {
                TL_CHECK_AND_PASS(m_outline.shaders[1]->set_texture(k_uniform_screen_texture, m_outline.render_buf->get_texture(GL_COLOR_ATTACHMENT0), 0));
                TL_CHECK_AND_PASS(m_outline.shaders[1]->set_uniform(k_uniform_outline_color, glm::vec3(1, 0, 0)));
                TL_CHECK_AND_PASS(m_outline.shaders[1]->set_uniform(k_uniform_thickness, 0.9f));
                TL_CHECK_AND_PASS(m_outline.shaders[1]->set_uniform(k_uniform_texel_size, 1.0f / glm::vec2(m_config.width, m_config.height)));
                TL_CHECK_AND_PASS(m_outline.quad_render_data->bind());
                TL_CHECK_AND_PASS(m_outline.quad_render_data->draw_array(GL_TRIANGLES));
                m_outline.quad_render_data->unbind();
            }
        }
        m_outline.render_buf->unbind();

        // attachment 1 now contains the outline
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (!fbo) {
            TL_CHECK_AND_PASS(MainFrameBuffer::bind());
        } else {
            TL_CHECK_AND_PASS(m_render_buf->bind());
        }

        TL_CHECK_AND_PASS(m_outline.shaders[2]->bind());
        {
            TL_CHECK_AND_PASS(m_outline.quad_render_data->bind());
            {
                TL_CHECK_AND_PASS(m_outline.shaders[2]->set_texture(k_uniform_screen_texture, m_outline.render_buf->get_texture(GL_COLOR_ATTACHMENT1), 0));
                TL_CHECK_AND_PASS(m_outline.quad_render_data->draw_array(GL_TRIANGLES));
            }
            m_outline.quad_render_data->unbind();
        }
        m_outline.shaders[2]->unbind();

        glDisable(GL_BLEND);

        return {};
    };

    if (!valid()) {
        return TL_ERROR( "invalid EditorRenderer");
    }

    if (!fbo) {
        TL_CHECK_AND_PASS(MainFrameBuffer::bind());
        TL_CHECK_AND_PASS(MainFrameBuffer::clear(k_clear_color, 1.0f));
    } else {
        TL_CHECK_AND_PASS(m_render_buf->bind());
        TL_CHECK_AND_PASS(m_render_buf->clear(k_clear_color, 1.0f));
    }

    glViewport(0, 0, static_cast<GLsizei>(get_config().width), static_cast<GLsizei>(get_config().height));

    // render objects
    for (auto [it, i] = std::tuple{ m_scene->begin(), 0 }; it != m_scene->end(); ++it) {
        auto obj = *it;
        auto res = try_get_obj_data(obj);
        if (!res) return TL_ERROR(res.error());

    	auto&& data = res.value().get();
        auto&& shader = data.shader;
        auto&& vao = data.render_data;

        TL_CHECK_AND_PASS(shader->bind());

        TL_CHECK_NON_FATAL(
            m_app, LogLevel::Warning,
            shader->set_uniform(k_uniform_light_pos, m_scene->get_good_light_pos())
        );
        TL_CHECK_NON_FATAL(
            m_app, LogLevel::Warning,
            shader->set_uniform(k_uniform_view, cam.get().get_view())
        );
        TL_CHECK_NON_FATAL(
            m_app, LogLevel::Warning,
            shader->set_uniform(k_uniform_projection, cam.get().get_projection())
        );
        TL_CHECK_NON_FATAL(
            m_app, LogLevel::Warning,
            shader->set_uniform(k_uniform_model, obj->get_transform().get_matrix())
        );

        TL_CHECK_AND_PASS(vao->bind());
        TL_CHECK_AND_PASS(vao->draw_array(GL_TRIANGLES));

        if (i == m_scene->size() - 1) {
            shader->unbind();
        }
    }

	// render grid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    TL_CHECK_AND_PASS(m_grid_render_data->bind());
    TL_CHECK_AND_PASS(m_grid_shader->bind());
    {
        TL_CHECK_AND_PASS(m_grid_shader->set_uniform(k_uniform_view, cam.get().get_view()));
        TL_CHECK_AND_PASS(m_grid_shader->set_uniform(k_uniform_projection, cam.get().get_projection()));
        TL_CHECK_AND_PASS(m_grid_render_data->draw_array(GL_LINES));
    }
    m_grid_shader->unbind();
    m_grid_render_data->unbind();

    glDisable(GL_BLEND);

    // render outline
    if (m_cur_outline_obj) {
        TL_CHECK_AND_PASS(draw_outline(m_cur_outline_obj));
    }

    // reset state
    if (fbo) {
        TL_CHECK_AND_PASS(MainFrameBuffer::bind());
    }
    return {};
}

void EditorRenderer::clear_render_data() {
    m_obj_data.clear();
}

void EditorRenderer::commit_cur_shader_code() noexcept {
    if (m_cur_outline_obj) {
        for (auto const shader_type : EIter<ShaderType>{}) {
            m_obj_data[m_cur_outline_obj].editing_data.shader_srcs[shader_type] =
                m_shader_editor_data[shader_type].editor.GetText();
        }
    }
}

#pragma region Shader_Editing

auto EditorRenderer::draw_glsl_editor(ShaderType type, Ref<ShaderProgram> shader, PerTextEditorData& data) noexcept
-> tl::expected <void, std::string>
{
    auto try_compile = [this]() {
        std::vector<ShaderProgram::StageDesc<std::string_view>> srcs;
        commit_cur_shader_code();

        auto const& data = m_obj_data[m_cur_outline_obj];
        for (auto const shader_type : EIter<ShaderType>{}) {
            auto const& text = data.editing_data.shader_srcs[shader_type];
            // if it has text, we'll try to compile it
            if (!text.empty()) {
                srcs.emplace_back(shader_type, text);
            }
        }

        auto res = m_obj_data[m_cur_outline_obj].shader->try_recompile(srcs);
        if (!res) {
            m_app->log(LogLevel::Error, res.error());
        }
    };


    auto&& editor = data.editor;
    if (ImGui::Button("Compile")) {
        try_compile();
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(
        ImGui::CalcTextSize(data.cur_font_size_mul_str).x * 3.0f + 
        ImGui::GetStyle().FramePadding.x * 2.0f
    );

    if (ImGui::BeginCombo("Font Size", data.cur_font_size_mul_str)) {
        for (size_t i = 0; i < std::size(PerTextEditorData::font_size_mul_strs); ++i)
        {
            auto const selected = data.cur_font_size_mul_str == data.font_size_mul_strs[i];
            if (ImGui::Selectable(PerTextEditorData::font_size_mul_strs[i], selected)) {
                data.cur_font_size_mul_str = PerTextEditorData::font_size_mul_strs[i];
            }
        }
        ImGui::EndCombo();
    }

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Save"))
            {
                auto textToSave = editor.GetText();
                // TODO: save
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
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
            if (ImGui::MenuItem("Paste", "Ctrl-V", nullptr, !ro && ImGui::GetClipboardText() != nullptr)) {
                editor.Paste();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Select all", nullptr, nullptr))
                editor.SetSelection(TextEditor::Coordinates(), TextEditor::Coordinates(editor.GetTotalLines(), 0));

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            if (ImGui::MenuItem("Dark palette"))
                editor.SetPalette(TextEditor::GetDarkPalette());
            if (ImGui::MenuItem("Light palette"))
                editor.SetPalette(TextEditor::GetLightPalette());
            if (ImGui::MenuItem("Retro blue palette"))
                editor.SetPalette(TextEditor::GetRetroBluePalette());
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // draw uniforms
    auto const uniforms = shader.get().get_uniform_map();

    if(ImGui::CollapsingHeader("uniform variables", ImGuiTreeNodeFlags_DefaultOpen)) {
        if(ImGui::BeginChild("##", ImVec2(0, 200), true))  {
            for(auto const& [name, val] : uniforms.get()) {
                bool disabled = std::find(std::begin(k_built_in_uniforms), std::end(k_built_in_uniforms), name) 
                    != std::end(k_built_in_uniforms);
                
                ShaderVariable val_copy = val;
                ImGui::BeginDisabled(disabled);
                if(disabled) {
                    auto display_name = name + " (built-in)";
                    ImGui::ShaderVariableField(display_name.c_str(), val_copy);
                } else {
                    if (ImGui::ShaderVariableField(name.c_str(), val_copy)) {
                        TL_CHECK(shader.get().bind());
                        TL_CHECK_NON_FATAL(
                            m_app, LogLevel::Warning,
                            shader.get().set_uniform(name, val_copy)
                        );
                        shader.get().unbind();
                    }
                }
                ImGui::EndDisabled();
                ImGui::Separator();
                ImGui::Spacing();
            }
            ImGui::EndChild();
        }
    }

    if(ImGui::CollapsingHeader("Built-in Uniform Declarations", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginDisabled();
        for (auto const& [name, val] : uniforms.get()) {
            bool built_in = std::find(std::begin(k_built_in_uniforms), std::end(k_built_in_uniforms), name)
                != std::end(k_built_in_uniforms);
            if(built_in) {
                ImGui::Text("uniform %s %s;", val.get_type_str(), name.data());
            }
        }
        ImGui::EndDisabled();
    }

    // draw input & outputs
    ImGui::Separator();
    auto const cpos = editor.GetCursorPosition();
    ImGui::SameLine();
    ImGui::Text("%6d/%-6d %6d lines  | %s | %s | %s | %s", cpos.mLine + 1, cpos.mColumn + 1, editor.GetTotalLines(),
        editor.IsOverwrite() ? "Ovr" : "Ins",
        editor.CanUndo() ? "*" : " ",
        editor.GetLanguageDefinition().mName.c_str(), m_cur_outline_obj->get_name().data());

    auto const default_font = ImGui::GetFont();
    ImFont font{ *default_font };
    font.Scale = data.get_font_size_mul();

	ImGui::PushFont(&font);
    editor.Render(to_string(type));
    ImGui::PopFont();

    return {};
}

auto EditorRenderer::try_get_obj_data(ViewPtr<Object> obj) noexcept -> tl::expected<Ref<PerObjectData>, std::string> {
    auto const it = m_obj_data.find(obj);
    if (it == m_obj_data.end()) {
        return  TL_ERROR("obj not found in render data");
    }
    return it->second;
}

auto EditorRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
    TL_CHECK_AND_PASS(Renderer::draw_imgui());

    ImGui::Begin("Shader Editor", nullptr, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_MenuBar);
    ImGui::SetWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    ImGui::SetWindowPos(ImVec2(700, 700), ImGuiCond_FirstUseEver);

    if (!m_cur_outline_obj) {
        ImGui::Text("Please Select an object first");
    } else {
        if (ImGui::BeginTabBar("##tab")) {
            for(auto const shader_type : EIter<ShaderType>{}) {
                if (ImGui::BeginTabItem(to_string(shader_type))) {
                    auto res = try_get_obj_data(m_cur_outline_obj);
                    if (!res) return TL_ERROR(res.error());

                    draw_glsl_editor(shader_type, *(res.value().get().shader), m_shader_editor_data[shader_type]);
                    ImGui::EndTabItem();
                }
            }

            ImGui::EndTabBar();
        }
    }

    ImGui::End();
    return {};
}

#pragma endregion Shader_Editing
