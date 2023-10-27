#include "include/glslHelper.h"
#include "include/editorResources.h"

#include <regex>
#include <sstream>

auto GLSLHelper::preprocess(ShaderType type, std::string_view common_funcs, std::string_view main_src) -> std::string {
    return std::string{ k_default_shader_header[type] } + common_funcs.data() + "\n" + main_src.data();
}

auto GLSLHelper::get_in_out(ShaderType type, std::string_view src) -> GLSLInfo {
    GLSLInfo info;

    std::ostringstream osss[2];
    std::regex const pattern{ R"((in|out)\s+(\w+)\s+(\w+);)" };
    std::match_results<typename decltype(src)::const_iterator> matches{};

    for (auto start = src.cbegin(); 
        std::regex_search(start, src.cend(), matches, pattern);
        start = matches.suffix().first) 
    {
        auto var_type = matches[2].str();
        auto var_name = matches[3].str();
        auto& oss = matches[1] == "in" ? osss[0] : osss[1];
        oss << var_name << '(' << var_type << ")\n";
    }

    if (type == ShaderType::Vertex) {
        for (auto start = k_vertex_attributes_decl.cbegin();
            std::regex_search(start, k_vertex_attributes_decl.cend(), matches, pattern);
            start = matches.suffix().first) 
        {
            auto var_type = matches[2].str();
            auto var_name = matches[3].str();
            auto& oss = matches[1] == "in" ? osss[0] : osss[1];
            oss << var_name << '(' << var_type << ")\n";
        }
    }

    info.inputs = osss[0].str();
    info.outputs = osss[1].str();
    return info;
}