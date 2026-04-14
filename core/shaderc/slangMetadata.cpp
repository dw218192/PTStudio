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

// Metadata-header walker. Walks a linked `slang::ShaderReflection` and emits
// a C++ header with entry-point names, vertex layout, bind group layouts, and
// fragment output count. Layout entries are derived by dispatching on the
// parameter's `TypeReflection::Kind` — buffers, textures, samplers, and
// storage textures each produce the right WGPU BindGroupLayoutEntry shape.
//
// Dynamic offsets are driven by the `[DynamicBuffer]` Slang attribute on the
// variable declaration (registered as a builtin in slangRuntime). When the
// attribute is present on a ConstantBuffer binding, `hasDynamicOffset=true`
// is emitted on the layout entry.

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
    // Exactly one of these categories is populated; empty category strings are
    // omitted from the emitted entry.
    std::string buffer_type;  // e.g. WGPUBufferBindingType_Uniform
    bool has_dynamic_offset = false;
    size_t min_binding_size = 0;

    std::string texture_sample_type;  // e.g. WGPUTextureSampleType_Float
    std::string texture_view_dim;     // e.g. WGPUTextureViewDimension_2D
    bool texture_multisampled = false;

    std::string sampler_type;  // e.g. WGPUSamplerBindingType_Filtering

    std::string storage_texture_access;    // e.g. WGPUStorageTextureAccess_WriteOnly
    std::string storage_texture_view_dim;  // e.g. WGPUStorageTextureViewDimension_2D
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

// ── bind entry classification ──

const char* wgpu_view_dim_for_shape(SlangResourceShape shape) {
    SlangResourceShape base =
        static_cast<SlangResourceShape>(shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
    bool is_array = (shape & SLANG_TEXTURE_ARRAY_FLAG) != 0;
    switch (base) {
        case SLANG_TEXTURE_1D:
            return "WGPUTextureViewDimension_1D";
        case SLANG_TEXTURE_2D:
            return is_array ? "WGPUTextureViewDimension_2DArray" : "WGPUTextureViewDimension_2D";
        case SLANG_TEXTURE_3D:
            return "WGPUTextureViewDimension_3D";
        case SLANG_TEXTURE_CUBE:
            return is_array ? "WGPUTextureViewDimension_CubeArray"
                            : "WGPUTextureViewDimension_Cube";
        default:
            return "WGPUTextureViewDimension_2D";
    }
}

const char* wgpu_sample_type_for(TypeReflection* result_type) {
    if (!result_type) return "WGPUTextureSampleType_Float";
    auto kind = result_type->getKind();
    TypeReflection::ScalarType st = TypeReflection::ScalarType::None;
    if (kind == TypeReflection::Kind::Vector || kind == TypeReflection::Kind::Scalar) {
        st = result_type->getScalarType();
    }
    switch (st) {
        case TypeReflection::ScalarType::Int8:
        case TypeReflection::ScalarType::Int16:
        case TypeReflection::ScalarType::Int32:
        case TypeReflection::ScalarType::Int64:
            return "WGPUTextureSampleType_Sint";
        case TypeReflection::ScalarType::UInt8:
        case TypeReflection::ScalarType::UInt16:
        case TypeReflection::ScalarType::UInt32:
        case TypeReflection::ScalarType::UInt64:
            return "WGPUTextureSampleType_Uint";
        default:
            return "WGPUTextureSampleType_Float";
    }
}

bool has_dynamic_buffer_attr(slang::IGlobalSession* global_session,
                             VariableLayoutReflection* var_layout) {
    if (!global_session || !var_layout) return false;
    auto* var = var_layout->getVariable();
    if (!var) return false;
    // The attribute type name in Slang source is `DynamicBufferAttribute`; the
    // `Attribute` suffix is dropped when referenced as `[DynamicBuffer]`. Slang
    // exposes the attribute under the full type name in reflection.
    return var->findAttributeByName(reinterpret_cast<SlangSession*>(global_session),
                                    "DynamicBuffer") != nullptr ||
           var->findAttributeByName(reinterpret_cast<SlangSession*>(global_session),
                                    "DynamicBufferAttribute") != nullptr;
}

bool has_non_filterable_attr(slang::IGlobalSession* global_session,
                             VariableLayoutReflection* var_layout) {
    if (!global_session || !var_layout) return false;
    auto* var = var_layout->getVariable();
    if (!var) return false;
    return var->findAttributeByName(reinterpret_cast<SlangSession*>(global_session),
                                    "NonFilterable") != nullptr ||
           var->findAttributeByName(reinterpret_cast<SlangSession*>(global_session),
                                    "NonFilterableAttribute") != nullptr;
}

bool has_non_filtering_attr(slang::IGlobalSession* global_session,
                            VariableLayoutReflection* var_layout) {
    if (!global_session || !var_layout) return false;
    auto* var = var_layout->getVariable();
    if (!var) return false;
    return var->findAttributeByName(reinterpret_cast<SlangSession*>(global_session),
                                    "NonFiltering") != nullptr ||
           var->findAttributeByName(reinterpret_cast<SlangSession*>(global_session),
                                    "NonFilteringAttribute") != nullptr;
}

// Classify a descriptor-table binding into a BindEntry. Populates exactly one
// category group (buffer / texture / sampler / storage_texture).
void classify_bind_entry(slang::IGlobalSession* global_session,
                         VariableLayoutReflection* var_layout, BindEntry& out) {
    auto* tl = var_layout->getTypeLayout();
    if (!tl) return;
    auto kind = tl->getKind();

    switch (kind) {
        case TypeReflection::Kind::ConstantBuffer:
        case TypeReflection::Kind::ParameterBlock: {
            out.buffer_type = "Uniform";
            if (auto* evl = tl->getElementVarLayout()) {
                if (auto* etl = evl->getTypeLayout()) {
                    out.min_binding_size =
                        static_cast<size_t>(etl->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM));
                }
            }
            out.has_dynamic_offset = has_dynamic_buffer_attr(global_session, var_layout);
            return;
        }
        case TypeReflection::Kind::SamplerState: {
            out.sampler_type =
                has_non_filtering_attr(global_session, var_layout) ? "NonFiltering" : "Filtering";
            return;
        }
        case TypeReflection::Kind::ShaderStorageBuffer: {
            // HLSL-style StructuredBuffer sometimes surfaces as this kind.
            auto access = tl->getResourceAccess();
            out.buffer_type =
                (access == SLANG_RESOURCE_ACCESS_READ_WRITE) ? "Storage" : "ReadOnlyStorage";
            return;
        }
        case TypeReflection::Kind::Resource: {
            SlangResourceShape shape = tl->getResourceShape();
            SlangResourceShape base =
                static_cast<SlangResourceShape>(shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
            auto access = tl->getResourceAccess();

            if (base == SLANG_STRUCTURED_BUFFER || base == SLANG_BYTE_ADDRESS_BUFFER) {
                out.buffer_type =
                    (access == SLANG_RESOURCE_ACCESS_READ_WRITE) ? "Storage" : "ReadOnlyStorage";
                return;
            }
            // Texture binding.
            if (access == SLANG_RESOURCE_ACCESS_READ_WRITE ||
                access == SLANG_RESOURCE_ACCESS_WRITE) {
                // Storage texture. Format is not recoverable from reflection;
                // callers set it via WebGPU descriptor. Emit access + view dim.
                out.storage_texture_access =
                    (access == SLANG_RESOURCE_ACCESS_READ_WRITE) ? "ReadWrite" : "WriteOnly";
                out.storage_texture_view_dim = wgpu_view_dim_for_shape(shape);
                return;
            }
            out.texture_sample_type = wgpu_sample_type_for(tl->getResourceResultType());
            if (has_non_filterable_attr(global_session, var_layout) &&
                std::string_view(out.texture_sample_type) == "WGPUTextureSampleType_Float") {
                out.texture_sample_type = "WGPUTextureSampleType_UnfilterableFloat";
            }
            out.texture_view_dim = wgpu_view_dim_for_shape(shape);
            out.texture_multisampled = (shape & SLANG_TEXTURE_MULTISAMPLE_FLAG) != 0;
            return;
        }
        default:
            // Unknown descriptor kind — leave all category strings empty, which
            // will emit a stub entry. Callers must extend this switch when new
            // binding shapes appear in shaders.
            return;
    }
}

std::string visibility_for(ShaderReflection* r, IComponentType* linked, int target_index,
                           slang::ParameterCategory cat, unsigned space, unsigned index) {
    bool use_vertex = false;
    bool use_fragment = false;
    bool use_compute = false;
    SlangUInt n_eps = r->getEntryPointCount();
    for (SlangUInt i = 0; i < n_eps; ++i) {
        auto* ep = r->getEntryPointByIndex(i);
        SlangStage stage = ep->getStage();
        if (stage != SLANG_STAGE_VERTEX && stage != SLANG_STAGE_FRAGMENT &&
            stage != SLANG_STAGE_COMPUTE) {
            continue;
        }
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
        else if (stage == SLANG_STAGE_COMPUTE)
            use_compute = true;
    }
    std::string out;
    auto add = [&](const char* s) {
        if (!out.empty()) out += " | ";
        out += s;
    };
    if (use_vertex) add("WGPUShaderStage_Vertex");
    if (use_fragment) add("WGPUShaderStage_Fragment");
    if (use_compute) add("WGPUShaderStage_Compute");
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

std::string run_slang_metadata_header(slang::IGlobalSession* global_session,
                                      slang::ShaderReflection* reflection,
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
            classify_bind_entry(global_session, p, e);
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

    // ── Render header ──
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
    o << "\n";

    for (const auto& bg : bind_groups) {
        o << "// ── Bind Group " << bg.group
          << " ────────────────────────────────────────────────\n";
        o << "inline WGPUBindGroupLayout create_bind_group_layout_" << bg.group
          << "(WGPUDevice device) {\n";
        for (const auto& e : bg.entries) {
            const std::string pre = "    entry" + std::to_string(e.binding);
            o << "    WGPUBindGroupLayoutEntry entry" << e.binding
              << " = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;\n";
            o << pre << ".binding = " << e.binding << ";\n";
            o << pre << ".visibility = " << e.visibility << ";\n";
            if (!e.buffer_type.empty()) {
                o << pre << ".buffer.type = WGPUBufferBindingType_" << e.buffer_type << ";\n";
                if (e.has_dynamic_offset) {
                    o << pre << ".buffer.hasDynamicOffset = true;\n";
                }
                if (e.min_binding_size > 0) {
                    o << pre << ".buffer.minBindingSize = " << e.min_binding_size << ";\n";
                }
            } else if (!e.sampler_type.empty()) {
                o << pre << ".sampler.type = WGPUSamplerBindingType_" << e.sampler_type << ";\n";
            } else if (!e.texture_sample_type.empty()) {
                o << pre << ".texture.sampleType = " << e.texture_sample_type << ";\n";
                o << pre << ".texture.viewDimension = " << e.texture_view_dim << ";\n";
                if (e.texture_multisampled) {
                    o << pre << ".texture.multisampled = true;\n";
                }
            } else if (!e.storage_texture_access.empty()) {
                o << pre << ".storageTexture.access = WGPUStorageTextureAccess_"
                  << e.storage_texture_access << ";\n";
                // Format is not recoverable from reflection; caller must set it
                // before using this layout. Emit a placeholder so the header is
                // still valid C++.
                o << pre << ".storageTexture.format = WGPUTextureFormat_Undefined;\n";
                o << pre << ".storageTexture.viewDimension = "
                  << (e.storage_texture_view_dim.empty() ? "WGPUTextureViewDimension_2D"
                                                         : e.storage_texture_view_dim)
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
