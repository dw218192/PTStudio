#ifndef __EMSCRIPTEN__

#include <core/rendering/shaderc/slangMetadata.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Metadata-header walker. Mirrors the byte-exact output of the pre-refactor
// Jinja template at core/templates/shader_metadata.h.j2 + Python walker in
// tools/repo_tools/shader_codegen.py. Keep output byte-compat when extending;
// the generated headers are checked in under */generated/ and consumed
// directly by the C++ render passes.

namespace {

using namespace slang;

struct VertexAttr {
    std::string name;
    unsigned location = 0;
    std::string wgpu_format;
    unsigned byte_size = 0;
};

struct BindEntry {
    unsigned binding = 0;
    std::string visibility;
    std::string buffer_type;
    size_t min_binding_size = 0;
};

struct BindGroup {
    unsigned group = 0;
    std::vector<BindEntry> entries;
};

// ── slang type → WGPUVertexFormat ──

bool vertex_format_for(TypeReflection* t, std::string& format_out, unsigned& size_out) {
    if (!t) return false;
    auto kind = t->getKind();
    if (kind == TypeReflection::Kind::Scalar) {
        switch (t->getScalarType()) {
            case TypeReflection::ScalarType::Float32:
                format_out = "WGPUVertexFormat_Float32";
                size_out = 4;
                return true;
            case TypeReflection::ScalarType::Int32:
                format_out = "WGPUVertexFormat_Sint32";
                size_out = 4;
                return true;
            case TypeReflection::ScalarType::UInt32:
                format_out = "WGPUVertexFormat_Uint32";
                size_out = 4;
                return true;
            default:
                return false;
        }
    }
    if (kind == TypeReflection::Kind::Vector) {
        unsigned count = static_cast<unsigned>(t->getElementCount());
        auto st = t->getScalarType();
        const char* base = nullptr;
        unsigned elem_bytes = 4;
        switch (st) {
            case TypeReflection::ScalarType::Float32:
                base = "WGPUVertexFormat_Float32";
                break;
            case TypeReflection::ScalarType::Int32:
                base = "WGPUVertexFormat_Sint32";
                break;
            case TypeReflection::ScalarType::UInt32:
                base = "WGPUVertexFormat_Uint32";
                break;
            default:
                return false;
        }
        if (count < 2 || count > 4) return false;
        format_out = std::string(base) + "x" + std::to_string(count);
        size_out = elem_bytes * count;
        return true;
    }
    return false;
}

bool has_category(VariableLayoutReflection* v, slang::ParameterCategory target) {
    if (!v) return false;
    unsigned n = v->getCategoryCount();
    for (unsigned i = 0; i < n; ++i) {
        if (v->getCategoryByIndex(i) == target) return true;
    }
    return false;
}

// Pick first non-Uniform category; fall back to first.
slang::ParameterCategory primary_category(VariableLayoutReflection* v) {
    unsigned n = v->getCategoryCount();
    for (unsigned i = 0; i < n; ++i) {
        auto c = v->getCategoryByIndex(i);
        if (c != slang::ParameterCategory::Uniform) return c;
    }
    return n > 0 ? v->getCategoryByIndex(0) : slang::ParameterCategory::None;
}

// ── vertex attribute collection ──

void collect_vertex_attrs_from_var(VariableLayoutReflection* v, std::vector<VertexAttr>& out) {
    if (!v) return;
    if (!has_category(v, slang::ParameterCategory::VaryingInput)) return;
    auto* tl = v->getTypeLayout();
    auto* t = tl ? tl->getType() : nullptr;
    if (!t) return;
    if (t->getKind() == TypeReflection::Kind::Struct) {
        unsigned nf = tl->getFieldCount();
        for (unsigned i = 0; i < nf; ++i) {
            auto* f = tl->getFieldByIndex(i);
            if (!has_category(f, slang::ParameterCategory::VaryingInput)) continue;
            VertexAttr attr;
            attr.name = f->getName() ? f->getName() : "";
            attr.location = f->getBindingIndex();
            if (!vertex_format_for(f->getTypeLayout()->getType(), attr.wgpu_format,
                                   attr.byte_size)) {
                continue;
            }
            out.push_back(std::move(attr));
        }
    } else {
        VertexAttr attr;
        attr.name = v->getName() ? v->getName() : "";
        attr.location = v->getBindingIndex();
        if (!vertex_format_for(t, attr.wgpu_format, attr.byte_size)) return;
        out.push_back(std::move(attr));
    }
}

// ── bind group helpers ──

std::string buffer_type_name(TypeLayoutReflection* tl) {
    if (!tl) return "Uniform";
    auto kind = tl->getKind();
    if (kind == TypeReflection::Kind::ConstantBuffer ||
        kind == TypeReflection::Kind::ParameterBlock) {
        return "Uniform";
    }
    if (kind == TypeReflection::Kind::Resource) {
        SlangResourceShape shape = tl->getResourceShape();
        SlangResourceShape base =
            static_cast<SlangResourceShape>(shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        if (base == SLANG_STRUCTURED_BUFFER) {
            if (tl->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ_WRITE) {
                return "Storage";
            }
            return "ReadOnlyStorage";
        }
    }
    return "Uniform";
}

size_t min_binding_size(TypeLayoutReflection* tl) {
    if (!tl) return 0;
    auto kind = tl->getKind();
    if (kind == TypeReflection::Kind::ConstantBuffer ||
        kind == TypeReflection::Kind::ParameterBlock) {
        if (auto* evl = tl->getElementVarLayout()) {
            if (auto* etl = evl->getTypeLayout()) {
                return static_cast<size_t>(etl->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM));
            }
        }
    }
    return 0;
}

std::string visibility_for(ShaderReflection* r, IComponentType* linked, int target_index,
                           slang::ParameterCategory cat, unsigned space, unsigned index) {
    bool use_vertex = false;
    bool use_fragment = false;
    SlangUInt n_eps = r->getEntryPointCount();
    for (SlangUInt i = 0; i < n_eps; ++i) {
        auto* ep = r->getEntryPointByIndex(i);
        SlangStage stage = ep->getStage();
        if (stage != SLANG_STAGE_VERTEX && stage != SLANG_STAGE_FRAGMENT) continue;
        bool used = true;  // permissive default without a linked program
        if (linked) {
            Slang::ComPtr<IMetadata> meta;
            Slang::ComPtr<IBlob> diag;
            auto hr = linked->getEntryPointMetadata(static_cast<SlangInt>(i), target_index,
                                                    meta.writeRef(), diag.writeRef());
            if (SLANG_SUCCEEDED(hr) && meta) {
                bool is_used = false;
                auto hr2 = meta->isParameterLocationUsed(static_cast<SlangParameterCategory>(cat),
                                                         space, index, is_used);
                if (SLANG_SUCCEEDED(hr2)) {
                    used = is_used;
                }
            }
        }
        if (!used) continue;
        if (stage == SLANG_STAGE_VERTEX)
            use_vertex = true;
        else if (stage == SLANG_STAGE_FRAGMENT)
            use_fragment = true;
    }
    std::string out;
    if (use_vertex) out += "WGPUShaderStage_Vertex";
    if (use_fragment) {
        if (!out.empty()) out += " | ";
        out += "WGPUShaderStage_Fragment";
    }
    if (out.empty()) {
        out = "WGPUShaderStage_Vertex | WGPUShaderStage_Fragment";
    }
    return out;
}

unsigned fragment_output_count(EntryPointReflection* ep) {
    if (!ep) return 1;
    auto* result = ep->getResultVarLayout();
    if (!result) return 1;
    auto* tl = result->getTypeLayout();
    if (!tl) return 1;
    if (tl->getKind() == TypeReflection::Kind::Struct) {
        unsigned count = 0;
        unsigned n = tl->getFieldCount();
        for (unsigned i = 0; i < n; ++i) {
            auto* f = tl->getFieldByIndex(i);
            if (has_category(f, slang::ParameterCategory::VaryingOutput)) {
                count++;
            }
        }
        return count > 0 ? count : 1;
    }
    return 1;
}

}  // namespace

namespace pts::rendering {

std::string run_slang_metadata_header(slang::ShaderReflection* reflection,
                                      slang::IComponentType* linked, std::string_view ns,
                                      int target_index) {
    // Discover entry points.
    EntryPointReflection* vertex_ep = nullptr;
    EntryPointReflection* fragment_ep = nullptr;
    if (reflection) {
        SlangUInt n_eps = reflection->getEntryPointCount();
        for (SlangUInt i = 0; i < n_eps; ++i) {
            auto* ep = reflection->getEntryPointByIndex(i);
            switch (ep->getStage()) {
                case SLANG_STAGE_VERTEX:
                    if (!vertex_ep) vertex_ep = ep;
                    break;
                case SLANG_STAGE_FRAGMENT:
                    if (!fragment_ep) fragment_ep = ep;
                    break;
                default:
                    break;
            }
        }
    }

    std::string vertex_entry = vertex_ep ? vertex_ep->getName() : "vs_main";
    std::string fragment_entry = fragment_ep ? fragment_ep->getName() : "fs_main";

    // Vertex layout.
    std::vector<VertexAttr> vertex_attrs;
    if (vertex_ep) {
        unsigned n = vertex_ep->getParameterCount();
        for (unsigned i = 0; i < n; ++i) {
            collect_vertex_attrs_from_var(vertex_ep->getParameterByIndex(i), vertex_attrs);
        }
    }
    std::sort(vertex_attrs.begin(), vertex_attrs.end(),
              [](const VertexAttr& a, const VertexAttr& b) { return a.location < b.location; });

    // Bind groups.
    std::vector<BindGroup> bind_groups;
    if (reflection) {
        unsigned n_params = reflection->getParameterCount();
        for (unsigned i = 0; i < n_params; ++i) {
            auto* p = reflection->getParameterByIndex(i);
            auto cat = primary_category(p);
            if (cat != slang::ParameterCategory::DescriptorTableSlot) continue;
            BindEntry e;
            e.binding = p->getBindingIndex();
            unsigned group =
                static_cast<unsigned>(p->getBindingSpace(static_cast<SlangParameterCategory>(cat)));
            auto* tl = p->getTypeLayout();
            e.buffer_type = buffer_type_name(tl);
            e.min_binding_size = min_binding_size(tl);
            e.visibility = visibility_for(reflection, linked, target_index, cat, group, e.binding);

            BindGroup* bg = nullptr;
            for (auto& g : bind_groups) {
                if (g.group == group) {
                    bg = &g;
                    break;
                }
            }
            if (!bg) {
                bind_groups.push_back(BindGroup{group, {}});
                bg = &bind_groups.back();
            }
            bg->entries.push_back(std::move(e));
        }
        std::sort(bind_groups.begin(), bind_groups.end(),
                  [](const BindGroup& a, const BindGroup& b) { return a.group < b.group; });
        for (auto& bg : bind_groups) {
            std::sort(bg.entries.begin(), bg.entries.end(),
                      [](const BindEntry& a, const BindEntry& b) { return a.binding < b.binding; });
        }
    }

    unsigned color_count = fragment_output_count(fragment_ep);

    // ── Render header (byte-compat with shader_metadata.h.j2) ──
    std::ostringstream o;
    o << "#pragma once\n";
    o << "// Auto-generated by shader_codegen — DO NOT EDIT\n";
    o << "\n";
    o << "#include <core/rendering/webgpu/webgpu.h>\n";
    o << "#include <array>\n";
    o << "#include <cstdint>\n";
    o << "\n";
    o << "namespace " << ns << " {\n";
    o << "\n";
    o << "// ── Entry Points ────────────────────────────────────────────────────\n";
    o << "inline constexpr const char* k_vertex_entry = \"" << vertex_entry << "\";\n";
    o << "inline constexpr const char* k_fragment_entry = \"" << fragment_entry << "\";\n";
    o << "\n";

    if (!vertex_attrs.empty()) {
        unsigned stride = 0;
        for (const auto& a : vertex_attrs) stride += a.byte_size;
        o << "// ── Vertex Attributes ───────────────────────────────────────────────\n";
        o << "struct VertexLayout {\n";
        o << "    static constexpr uint64_t stride = " << stride << ";\n";
        o << "    static constexpr WGPUVertexStepMode step_mode = WGPUVertexStepMode_Vertex;\n";
        o << "    static constexpr std::array<WGPUVertexAttribute, " << vertex_attrs.size()
          << "> attributes = {{\n";
        unsigned offset = 0;
        for (const auto& a : vertex_attrs) {
            o << "        {nullptr, " << a.wgpu_format << ", " << offset << ", " << a.location
              << "},  // " << a.name << "\n";
            offset += a.byte_size;
        }
        o << "    }};\n";
        o << "};\n";
    }
    // Blank line always precedes the bind-group section (template has a
    // literal blank line between the `{% endif %}` and the `{% for bg %}`).
    o << "\n";

    for (const auto& bg : bind_groups) {
        o << "// ── Bind Group " << bg.group
          << " ────────────────────────────────────────────────\n";
        o << "inline WGPUBindGroupLayout create_bind_group_layout_" << bg.group
          << "(WGPUDevice device) {\n";
        for (const auto& e : bg.entries) {
            o << "    WGPUBindGroupLayoutEntry entry" << e.binding
              << " = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;\n";
            o << "    entry" << e.binding << ".binding = " << e.binding << ";\n";
            o << "    entry" << e.binding << ".visibility = " << e.visibility << ";\n";
            o << "    entry" << e.binding << ".buffer.type = WGPUBufferBindingType_"
              << e.buffer_type << ";\n";
            if (e.min_binding_size > 0) {
                o << "    entry" << e.binding << ".buffer.minBindingSize = " << e.min_binding_size
                  << ";\n";
            }
            o << "\n";
        }
        if (bg.entries.size() == 1) {
            o << "    WGPUBindGroupLayoutDescriptor desc = "
                 "WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;\n";
            o << "    desc.entryCount = 1;\n";
            o << "    desc.entries = &entry" << bg.entries[0].binding << ";\n";
        } else {
            o << "    WGPUBindGroupLayoutEntry entries[] = {\n";
            for (const auto& e : bg.entries) {
                o << "        entry" << e.binding << ",\n";
            }
            o << "    };\n";
            o << "    WGPUBindGroupLayoutDescriptor desc = "
                 "WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;\n";
            o << "    desc.entryCount = " << bg.entries.size() << ";\n";
            o << "    desc.entries = entries;\n";
        }
        o << "    return wgpuDeviceCreateBindGroupLayout(device, &desc);\n";
        o << "}\n";
        o << "\n";
    }

    o << "// ── Fragment Outputs ────────────────────────────────────────────────\n";
    o << "inline constexpr uint32_t k_color_attachment_count = " << color_count << ";\n";
    o << "\n";
    o << "}  // namespace " << ns << "\n";

    return o.str();
}

}  // namespace pts::rendering

#endif  // !__EMSCRIPTEN__
