#pragma once

constexpr char const* k_imgui_ini = 
"\n\
[Window][Debug##Default]\n\
ViewportPos=10,10\n\
ViewportId=0x9F5F46A1\n\
Size=400,400\n\
Collapsed=0\n\
\n\
[Window][Scene Settings]\n\
Pos=0,0\n\
Size=187,542\n\
Collapsed=0\n\
DockId=0x00000002,0\n\
\n\
[Window][DockSpaceViewport_11111111]\n\
Pos=0,0\n\
Size=1280,720\n\
Collapsed=0\n\
\n\
[Window][Scene]\n\
Pos=189,0\n\
Size=880,542\n\
Collapsed=0\n\
DockId=0x00000003,0\n\
\n\
[Window][Inspector]\n\
Pos=1071,0\n\
Size=209,542\n\
Collapsed=0\n\
DockId=0x00000004,0\n\
\n\
[Window][Console]\n\
Pos=0,544\n\
Size=1280,176\n\
Collapsed=0\n\
DockId=0x00000006,0\n\
\n\
[Docking][Data]\n\
DockSpace       ID=0x8B93E3BD Window=0xA787BDB4 Pos=164,187 Size=1280,720 Split=Y Selected=0xE192E354\n\
  DockNode      ID=0x00000005 Parent=0x8B93E3BD SizeRef=1280,542 Split=X\n\
    DockNode    ID=0x00000001 Parent=0x00000005 SizeRef=1069,720 Split=X\n\
      DockNode  ID=0x00000002 Parent=0x00000001 SizeRef=187,720 Selected=0xD4E24632\n\
      DockNode  ID=0x00000003 Parent=0x00000001 SizeRef=880,720 CentralNode=1 Selected=0xE192E354\n\
    DockNode    ID=0x00000004 Parent=0x00000005 SizeRef=209,720 Selected=0xE7039252\n\
  DockNode      ID=0x00000006 Parent=0x8B93E3BD SizeRef=1280,176 Selected=0x49278EEE\n\
\n\
";

// shaders

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

constexpr char const* vs_grid_src = 
"\
    #version 330 core\n\
    layout (location = 0) in vec3 aPos;\n\
    uniform mat4 view;\n\
    uniform mat4 projection;\n\
    out vec2 grid_coords;\n\
    void main() {\n\
        grid_coords = aPos.xz;\n\
        gl_Position = projection * view * vec4(aPos, 1.0);\n\
    }\n\
";

constexpr char const* ps_grid_src = 
"\
    #version 330 core\n\
    uniform float half_grid_dim;\n\
    in vec2 grid_coords;\n\
    out vec4 FragColor;\n\
    void main() {\n\
        float dist = max(abs(grid_coords.x), abs(grid_coords.y)) / half_grid_dim;\n\
        float alpha = 1.0 - pow(dist, 0.25);\n\
        FragColor = vec4(0.7, 0.7, 0.7, alpha);\n\
    }\n\
";

constexpr char const* k_uniform_model = "model";
constexpr char const* k_uniform_view = "view";
constexpr char const* k_uniform_projection = "projection";
constexpr char const* k_uniform_light_pos = "lightPos";
constexpr char const* k_uniform_light_color = "lightColor";
constexpr char const* k_uniform_object_color = "objectColor";
constexpr char const* k_uniform_half_grid_dim = "half_grid_dim";