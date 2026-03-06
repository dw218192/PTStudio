#include <slang.h>
#include <slang-com-ptr.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace slang;

// ── CLI parsing ─────────────────────────────────────────────────────────────

struct EntryPointArg {
    std::string name;
    SlangStage stage;
};

struct Args {
    std::string input;
    std::string output;
    std::string ns;
    std::vector<EntryPointArg> entry_points;
};

static SlangStage parse_stage(const std::string& s) {
    if (s == "vertex") return SLANG_STAGE_VERTEX;
    if (s == "fragment") return SLANG_STAGE_FRAGMENT;
    if (s == "compute") return SLANG_STAGE_COMPUTE;
    fprintf(stderr, "error: unknown stage '%s'\n", s.c_str());
    exit(1);
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            args.ns = argv[++i];
        } else if (a == "-e" && i + 1 < argc) {
            std::string val = argv[++i];
            auto colon = val.find(':');
            if (colon == std::string::npos) {
                fprintf(stderr, "error: -e expects name:stage\n");
                exit(1);
            }
            EntryPointArg ep;
            ep.name = val.substr(0, colon);
            ep.stage = parse_stage(val.substr(colon + 1));
            args.entry_points.push_back(ep);
        } else if (a[0] != '-') {
            args.input = a;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            exit(1);
        }
    }
    if (args.input.empty() || args.output.empty() || args.ns.empty() || args.entry_points.empty()) {
        fprintf(stderr, "usage: slang_reflect <input.slang> -o <output.h> -n <namespace> -e <name:stage> [-e ...]\n");
        exit(1);
    }
    return args;
}

// ── Type mapping ────────────────────────────────────────────────────────────

struct VertexFormatInfo {
    const char* format;
    uint64_t size;
};

static VertexFormatInfo get_vertex_format(TypeReflection::ScalarType scalar, unsigned count) {
    if (scalar == TypeReflection::ScalarType::Float32) {
        switch (count) {
        case 1: return {"WGPUVertexFormat_Float32", 4};
        case 2: return {"WGPUVertexFormat_Float32x2", 8};
        case 3: return {"WGPUVertexFormat_Float32x3", 12};
        case 4: return {"WGPUVertexFormat_Float32x4", 16};
        }
    } else if (scalar == TypeReflection::ScalarType::Int32) {
        switch (count) {
        case 1: return {"WGPUVertexFormat_Sint32", 4};
        case 2: return {"WGPUVertexFormat_Sint32x2", 8};
        case 3: return {"WGPUVertexFormat_Sint32x3", 12};
        case 4: return {"WGPUVertexFormat_Sint32x4", 16};
        }
    } else if (scalar == TypeReflection::ScalarType::UInt32) {
        switch (count) {
        case 1: return {"WGPUVertexFormat_Uint32", 4};
        case 2: return {"WGPUVertexFormat_Uint32x2", 8};
        case 3: return {"WGPUVertexFormat_Uint32x3", 12};
        case 4: return {"WGPUVertexFormat_Uint32x4", 16};
        }
    }
    fprintf(stderr, "error: unsupported vertex type (scalar=%d, count=%u)\n", scalar, count);
    exit(1);
}

// ── Vertex attribute extraction ─────────────────────────────────────────────

struct VertexAttr {
    unsigned location;
    std::string name;
    const char* format;
    uint64_t size;
};

static VertexFormatInfo type_to_vertex_format(TypeLayoutReflection* typeLayout) {
    auto* type = typeLayout->getType();
    auto kind = type->getKind();

    if (kind == TypeReflection::Kind::Scalar) {
        return get_vertex_format(type->getScalarType(), 1);
    }
    if (kind == TypeReflection::Kind::Vector) {
        auto scalar = type->getScalarType();
        auto count = (unsigned)type->getColumnCount();
        return get_vertex_format(scalar, count);
    }
    fprintf(stderr, "error: unsupported vertex attribute type kind %d\n", (int)kind);
    exit(1);
}

static std::vector<VertexAttr> extract_vertex_attrs(EntryPointReflection* ep) {
    std::vector<VertexAttr> attrs;
    unsigned paramCount = ep->getParameterCount();
    for (unsigned i = 0; i < paramCount; ++i) {
        auto* param = ep->getParameterByIndex(i);
        if (param->getCategory() != ParameterCategory::VaryingInput)
            continue;

        auto* typeLayout = param->getTypeLayout();
        auto kind = typeLayout->getType()->getKind();

        if (kind == TypeReflection::Kind::Struct) {
            unsigned fieldCount = typeLayout->getFieldCount();
            for (unsigned j = 0; j < fieldCount; ++j) {
                auto* field = typeLayout->getFieldByIndex(j);
                if (field->getCategory() != ParameterCategory::VaryingInput)
                    continue;
                auto fmt = type_to_vertex_format(field->getTypeLayout());
                VertexAttr a;
                a.location = field->getBindingIndex();
                a.name = field->getVariable()->getName();
                a.format = fmt.format;
                a.size = fmt.size;
                attrs.push_back(a);
            }
        } else {
            auto fmt = type_to_vertex_format(typeLayout);
            VertexAttr a;
            a.location = param->getBindingIndex();
            a.name = param->getName() ? param->getName() : "";
            a.format = fmt.format;
            a.size = fmt.size;
            attrs.push_back(a);
        }
    }
    std::sort(attrs.begin(), attrs.end(),
              [](const VertexAttr& a, const VertexAttr& b) { return a.location < b.location; });
    return attrs;
}

// ── Bind group extraction ───────────────────────────────────────────────────

struct BindEntry {
    unsigned binding;
    std::string visibility;
    std::string buffer_type;
    uint64_t min_binding_size;
};

struct BindGroup {
    unsigned group;
    std::vector<BindEntry> entries;
};

static const char* stage_to_visibility(SlangStage stage) {
    switch (stage) {
    case SLANG_STAGE_VERTEX: return "WGPUShaderStage_Vertex";
    case SLANG_STAGE_FRAGMENT: return "WGPUShaderStage_Fragment";
    case SLANG_STAGE_COMPUTE: return "WGPUShaderStage_Compute";
    default: return "WGPUShaderStage_Vertex | WGPUShaderStage_Fragment";
    }
}

static std::string buffer_type_from_kind(TypeReflection::Kind kind) {
    switch (kind) {
    case TypeReflection::Kind::ConstantBuffer: return "Uniform";
    case TypeReflection::Kind::ShaderStorageBuffer: return "Storage";
    case TypeReflection::Kind::Resource: return "ReadOnlyStorage";
    default: return "Uniform";
    }
}

static bool binding_used_in_code(IComponentType* linked, unsigned target_binding, unsigned target_space) {
    Slang::ComPtr<IBlob> code;
    Slang::ComPtr<IBlob> diag;
    if (SLANG_FAILED(linked->getEntryPointCode(0, 0, code.writeRef(), diag.writeRef())))
        return false;
    if (!code) return false;

    std::string wgsl((const char*)code->getBufferPointer(), code->getBufferSize());
    // Look for @binding(N) in the WGSL output — if the binding appears, the EP uses it
    std::string pattern = "@binding(" + std::to_string(target_binding) + ")";
    return wgsl.find(pattern) != std::string::npos;
}

static std::vector<BindGroup> extract_bind_groups(
    ProgramLayout* fullLayout,
    ISession* session,
    IModule* module,
    const std::vector<Slang::ComPtr<IEntryPoint>>& entryPoints,
    const std::vector<EntryPointArg>& epArgs) {

    // Map: group -> binding -> BindEntry
    std::map<unsigned, std::map<unsigned, BindEntry>> groups;

    unsigned paramCount = fullLayout->getParameterCount();
    for (unsigned i = 0; i < paramCount; ++i) {
        auto* param = fullLayout->getParameterByIndex(i);

        // Check if parameter has DescriptorTableSlot category
        unsigned catCount = param->getCategoryCount();
        bool has_dts = false;
        for (unsigned c = 0; c < catCount; ++c) {
            if (param->getCategoryByIndex(c) == ParameterCategory::DescriptorTableSlot) {
                has_dts = true;
                break;
            }
        }
        if (!has_dts) continue;

        unsigned binding = param->getBindingIndex();
        unsigned space = (unsigned)param->getBindingSpace(ParameterCategory::DescriptorTableSlot);

        auto* typeLayout = param->getTypeLayout();
        auto typeKind = typeLayout->getType()->getKind();
        auto bufType = buffer_type_from_kind(typeKind);

        uint64_t minSize = 0;
        if (typeKind == TypeReflection::Kind::ConstantBuffer ||
            typeKind == TypeReflection::Kind::ShaderStorageBuffer) {
            auto* elementLayout = typeLayout->getElementTypeLayout();
            if (elementLayout) {
                minSize = elementLayout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
            }
        }

        // Determine visibility by checking generated WGSL code for each entry point
        std::vector<std::string> visFlags;
        for (size_t ep_i = 0; ep_i < entryPoints.size(); ++ep_i) {
            IComponentType* components[] = {module, entryPoints[ep_i]};
            Slang::ComPtr<IComponentType> composite;
            Slang::ComPtr<ISlangBlob> diag;
            if (SLANG_FAILED(session->createCompositeComponentType(
                    components, 2, composite.writeRef(), diag.writeRef())))
                continue;

            Slang::ComPtr<IComponentType> linked;
            if (SLANG_FAILED(composite->link(linked.writeRef(), diag.writeRef())))
                continue;

            if (binding_used_in_code(linked, binding, space)) {
                visFlags.push_back(stage_to_visibility(epArgs[ep_i].stage));
            }
        }

        std::string visibility;
        if (visFlags.empty()) {
            visibility = "WGPUShaderStage_Vertex | WGPUShaderStage_Fragment";
        } else {
            for (size_t v = 0; v < visFlags.size(); ++v) {
                if (v > 0) visibility += " | ";
                visibility += visFlags[v];
            }
        }

        BindEntry entry;
        entry.binding = binding;
        entry.visibility = visibility;
        entry.buffer_type = bufType;
        entry.min_binding_size = minSize;
        groups[space][binding] = entry;
    }

    std::vector<BindGroup> result;
    for (auto& [group_num, bindings] : groups) {
        BindGroup bg;
        bg.group = group_num;
        for (auto& [_, entry] : bindings) {
            bg.entries.push_back(entry);
        }
        result.push_back(bg);
    }
    return result;
}

// ── Fragment output count ───────────────────────────────────────────────────

static unsigned count_fragment_outputs(EntryPointReflection* ep) {
    auto* resultLayout = ep->getResultVarLayout();
    if (!resultLayout) return 1;

    auto* typeLayout = resultLayout->getTypeLayout();
    if (!typeLayout) return 1;

    if (typeLayout->getType()->getKind() == TypeReflection::Kind::Struct) {
        unsigned count = 0;
        unsigned fieldCount = typeLayout->getFieldCount();
        for (unsigned i = 0; i < fieldCount; ++i) {
            auto* field = typeLayout->getFieldByIndex(i);
            if (field->getCategory() == ParameterCategory::VaryingOutput)
                ++count;
        }
        return count > 0 ? count : 1;
    }
    return 1;
}

// ── Code generation ─────────────────────────────────────────────────────────

static std::string make_section_comment(const std::string& title) {
    // "// ── Title ──────...────" padded to ~72 display characters
    // ── is U+2500 BOX DRAWINGS LIGHT HORIZONTAL (3 bytes each in UTF-8)
    std::string prefix = "// \xe2\x94\x80\xe2\x94\x80 " + title + " ";
    // Count display chars: "// " (3) + "──" (2) + " " (1) + title + " " (1) = 7 + title.size()
    size_t display_len = 7 + title.size();
    std::string result = prefix;
    while (display_len < 71) {
        result += "\xe2\x94\x80";
        ++display_len;
    }
    return result;
}

static void emit_header(std::ostream& out, const Args& args,
                         const std::vector<VertexAttr>& vertexAttrs,
                         const std::vector<BindGroup>& bindGroups,
                         unsigned colorAttachmentCount,
                         const std::string& vertexEntry,
                         const std::string& fragmentEntry) {
    out << "#pragma once\n";
    out << "// Auto-generated by slang_reflect \xe2\x80\x94 DO NOT EDIT\n";
    out << "\n";
    out << "#include <core/rendering/webgpu/webgpu.h>\n";
    out << "#include <array>\n";
    out << "#include <cstdint>\n";
    out << "\n";
    out << "namespace " << args.ns << " {\n";
    out << "\n";

    // Entry points
    out << make_section_comment("Entry Points") << "\n";
    out << "inline constexpr const char* k_vertex_entry = \"" << vertexEntry << "\";\n";
    out << "inline constexpr const char* k_fragment_entry = \"" << fragmentEntry << "\";\n";

    // Vertex attributes
    if (!vertexAttrs.empty()) {
        out << "\n";
        out << make_section_comment("Vertex Attributes") << "\n";
        out << "struct VertexLayout {\n";

        uint64_t stride = 0;
        for (auto& a : vertexAttrs) stride += a.size;

        out << "    static constexpr uint64_t stride = " << stride << ";\n";
        out << "    static constexpr WGPUVertexStepMode step_mode = WGPUVertexStepMode_Vertex;\n";
        out << "    static constexpr std::array<WGPUVertexAttribute, " << vertexAttrs.size() << "> attributes = {{\n";

        uint64_t offset = 0;
        for (auto& a : vertexAttrs) {
            out << "        {nullptr, " << a.format << ", " << offset << ", " << a.location << "},  // " << a.name << "\n";
            offset += a.size;
        }
        out << "    }};\n";
        out << "};\n";
    }

    // Bind groups
    for (auto& bg : bindGroups) {
        out << "\n";
        out << make_section_comment("Bind Group " + std::to_string(bg.group)) << "\n";
        out << "inline WGPUBindGroupLayout create_bind_group_layout_" << bg.group << "(WGPUDevice device) {\n";

        for (auto& e : bg.entries) {
            out << "    WGPUBindGroupLayoutEntry entry" << e.binding << " = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;\n";
            out << "    entry" << e.binding << ".binding = " << e.binding << ";\n";
            out << "    entry" << e.binding << ".visibility = " << e.visibility << ";\n";
            out << "    entry" << e.binding << ".buffer.type = WGPUBufferBindingType_" << e.buffer_type << ";\n";
            if (e.min_binding_size > 0) {
                out << "    entry" << e.binding << ".buffer.minBindingSize = " << e.min_binding_size << ";\n";
            }
            out << "\n";
        }

        if (bg.entries.size() == 1) {
            out << "    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;\n";
            out << "    desc.entryCount = 1;\n";
            out << "    desc.entries = &entry" << bg.entries[0].binding << ";\n";
        } else {
            out << "    WGPUBindGroupLayoutEntry entries[] = {\n";
            for (auto& e : bg.entries) {
                out << "        entry" << e.binding << ",\n";
            }
            out << "    };\n";
            out << "    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;\n";
            out << "    desc.entryCount = " << bg.entries.size() << ";\n";
            out << "    desc.entries = entries;\n";
        }
        out << "    return wgpuDeviceCreateBindGroupLayout(device, &desc);\n";
        out << "}\n";
    }

    // Fragment outputs
    out << "\n";
    out << make_section_comment("Fragment Outputs") << "\n";
    out << "inline constexpr uint32_t k_color_attachment_count = " << colorAttachmentCount << ";\n";
    out << "\n";
    out << "}  // namespace " << args.ns << "\n";
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    // Extract module name (stem) and search path from input path
    std::string inputPath = args.input;
    std::string searchPath;
    std::string moduleName;
    {
        // Normalize to forward slashes
        std::string path = inputPath;
        for (auto& c : path) {
            if (c == '\\') c = '/';
        }

        auto lastSlash = path.rfind('/');
        if (lastSlash != std::string::npos) {
            searchPath = path.substr(0, lastSlash);
            moduleName = path.substr(lastSlash + 1);
        } else {
            searchPath = ".";
            moduleName = path;
        }

        // Strip .slang extension
        auto dot = moduleName.rfind('.');
        if (dot != std::string::npos) {
            moduleName = moduleName.substr(0, dot);
        }
    }

    // 1. Create global session
    Slang::ComPtr<IGlobalSession> globalSession;
    if (SLANG_FAILED(createGlobalSession(globalSession.writeRef()))) {
        fprintf(stderr, "error: failed to create Slang global session\n");
        return 1;
    }

    // 2. Set up target (WGSL) and session
    TargetDesc targetDesc = {};
    targetDesc.structureSize = sizeof(targetDesc);
    targetDesc.format = SLANG_WGSL;

    SessionDesc sessionDesc = {};
    sessionDesc.structureSize = sizeof(sessionDesc);
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    const char* searchPaths[] = {searchPath.c_str()};
    sessionDesc.searchPaths = searchPaths;
    sessionDesc.searchPathCount = 1;

    Slang::ComPtr<ISession> session;
    if (SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef()))) {
        fprintf(stderr, "error: failed to create Slang session\n");
        return 1;
    }

    // 3. Load module
    Slang::ComPtr<ISlangBlob> diagnostics;
    IModule* module = session->loadModule(moduleName.c_str(), diagnostics.writeRef());
    if (!module) {
        if (diagnostics) {
            fprintf(stderr, "error loading module '%s': %s\n",
                    moduleName.c_str(), (const char*)diagnostics->getBufferPointer());
        }
        return 1;
    }

    // 4. Find entry points
    std::vector<Slang::ComPtr<IEntryPoint>> entryPoints;
    for (auto& epArg : args.entry_points) {
        Slang::ComPtr<IEntryPoint> ep;
        if (SLANG_FAILED(module->findEntryPointByName(epArg.name.c_str(), ep.writeRef()))) {
            fprintf(stderr, "error: entry point '%s' not found\n", epArg.name.c_str());
            return 1;
        }
        entryPoints.push_back(ep);
    }

    // 5. Create composite [module, ep0, ep1, ...]
    std::vector<IComponentType*> components;
    components.push_back(module);
    for (auto& ep : entryPoints) {
        components.push_back(ep);
    }

    Slang::ComPtr<IComponentType> composite;
    if (SLANG_FAILED(session->createCompositeComponentType(
            components.data(), (SlangInt)components.size(),
            composite.writeRef(), diagnostics.writeRef()))) {
        fprintf(stderr, "error: failed to create composite component\n");
        return 1;
    }

    // 6. Link
    Slang::ComPtr<IComponentType> linked;
    if (SLANG_FAILED(composite->link(linked.writeRef(), diagnostics.writeRef()))) {
        fprintf(stderr, "error: failed to link program\n");
        if (diagnostics) {
            fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
        }
        return 1;
    }

    // 7. Get layout
    ProgramLayout* layout = linked->getLayout(0, diagnostics.writeRef());
    if (!layout) {
        fprintf(stderr, "error: failed to get program layout\n");
        return 1;
    }

    // 8. Extract reflection data
    std::string vertexEntry, fragmentEntry;
    std::vector<VertexAttr> vertexAttrs;
    unsigned colorAttachmentCount = 1;

    SlangUInt epCount = layout->getEntryPointCount();
    for (SlangUInt i = 0; i < epCount; ++i) {
        auto* ep = layout->getEntryPointByIndex(i);
        auto stage = ep->getStage();
        if (stage == SLANG_STAGE_VERTEX) {
            vertexEntry = ep->getName();
            vertexAttrs = extract_vertex_attrs(ep);
        } else if (stage == SLANG_STAGE_FRAGMENT) {
            fragmentEntry = ep->getName();
            colorAttachmentCount = count_fragment_outputs(ep);
        }
    }

    auto bindGroups = extract_bind_groups(layout, session, module, entryPoints, args.entry_points);

    // 9. Write output
    std::ofstream outFile(args.output);
    if (!outFile) {
        fprintf(stderr, "error: cannot open output file '%s'\n", args.output.c_str());
        return 1;
    }

    emit_header(outFile, args, vertexAttrs, bindGroups, colorAttachmentCount,
                vertexEntry, fragmentEntry);

    fprintf(stderr, "slang_reflect: wrote %s\n", args.output.c_str());
    return 0;
}
