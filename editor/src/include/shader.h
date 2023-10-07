#pragma once

#include <GL/glew.h>
#include <string>
#include <string_view>

#include "result.h"
#include "ext.h"

constexpr char const* ps_unicolor_src =
"\
	#version 330 core\n\
	layout(location = 0) out vec3 FragColor; \n\
	void main() {\n\
        FragColor = vec3(1.0f, 0.5f, 0.2f); \n\
	}\n\
";

constexpr char const* vs_obj_src =
"\
    #version 330 core\n\
    layout (location = 0) in vec3 aPos;\n\
    layout (location = 1) in vec3 aNormal;\n\
    layout (location = 2) in vec2 aTexCoords;\n\
    out vec2 TexCoords;\n\
    out vec3 Normal;\n\
    out vec3 FragPos;\n\
    uniform mat4 model;\n\
    uniform mat4 view;\n\
    uniform mat4 projection;\n\
    void main() {\n\
        TexCoords = aTexCoords;\n\
        Normal = mat3(transpose(inverse(model))) * aNormal;\n\
        FragPos = vec3(model * vec4(aPos, 1.0));\n\
        gl_Position = projection * view * vec4(FragPos, 1.0);\n\
    }\n\
";

constexpr char const* ps_obj_src =
"\
    #version 330 core\n\
    out vec4 FragColor;\n\
    in vec2 TexCoords;\n\
    in vec3 Normal;\n\
    in vec3 FragPos;\n\
	uniform vec3 lightPos;\n\
    uniform mat4 view;\n\
    void main() {\n\
        const vec3 objectColor = vec3(178.0/255.0, 190.0/255.0, 181.0/255.0);\n\
        const vec3 lightColor = vec3(1.0, 1.0, 1.0);\n\
		vec3 camPos = view[3].xyz;\n\
        float ambientStrength = 0.2;\n\
        vec3 ambient = ambientStrength * lightColor;\n\
        vec3 norm = normalize(Normal);\n\
        vec3 lightDir = normalize(lightPos - FragPos);\n\
        float diff = max(dot(norm, lightDir), 0.0);\n\
        vec3 diffuse = diff * lightColor;\n\
        float specularStrength = 0.5;\n\
        vec3 viewDir = normalize(camPos - FragPos);\n\
        vec3 reflectDir = reflect(-lightDir, norm);\n\
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);\n\
        vec3 specular = specularStrength * spec * lightColor;\n\
        vec3 result = (ambient + diffuse + specular) * objectColor;\n\
        FragColor = vec4(result, 1.0);\n\
    }\n\
";

constexpr char const* k_uniform_model = "model";
constexpr char const* k_uniform_view = "view";
constexpr char const* k_uniform_projection = "projection";
constexpr char const* k_uniform_light_pos = "lightPos";
constexpr char const* k_uniform_light_color = "lightColor";
constexpr char const* k_uniform_object_color = "objectColor";

enum class ShaderType {
    Vertex,
    Fragment
};

struct Shader {
    friend struct ShaderProgram;
    explicit Shader(ShaderType type) noexcept;
    ~Shader() noexcept;

    Shader(Shader&&) noexcept;
    Shader& operator=(Shader&&) noexcept;

    // shouldn't be copied because we have handles to GL resources
    Shader(Shader&) = delete;
    Shader& operator=(Shader&) = delete;

    [[nodiscard]] auto from_file(std::string_view file) noexcept -> Result<void>;
    [[nodiscard]] auto from_src(std::string_view src) noexcept -> Result<void>;
    [[nodiscard]] auto valid() const noexcept -> bool { return m_handle != 0; }

private:
    Shader() noexcept : m_type{0}, m_handle(0) { }
    GLenum m_type;
    GLuint m_handle;
};

struct ShaderProgram {
    ShaderProgram() noexcept : m_handle(0) { }
    ~ShaderProgram() noexcept;

    ShaderProgram(ShaderProgram&&) noexcept;
    ShaderProgram& operator=(ShaderProgram&&) noexcept;

	// shouldn't be copied because we have handles to GL resources
    ShaderProgram(ShaderProgram&) = delete;
    ShaderProgram& operator=(ShaderProgram&) = delete;

    [[nodiscard]] auto from_shaders(Shader&& vs, Shader&& ps) noexcept -> Result<void>;
    [[nodiscard]] auto set_uniform(std::string_view name, glm::mat4 const& value) const noexcept -> Result<void>;
    [[nodiscard]] auto set_uniform(std::string_view name, glm::vec3 const& value) const noexcept -> Result<void>;
    
    [[nodiscard]] auto valid() const noexcept { return m_handle != 0; }

    void use() const noexcept;
    void unuse() const noexcept;
private:
    GLuint m_handle;
    Shader m_vs;
    Shader m_ps;
};