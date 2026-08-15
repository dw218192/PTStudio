#ifndef __EMSCRIPTEN__

#include <core/rendering/shaderc/slangMetadata.h>
#include <core/rendering/shaderc/slangRuntime.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include <mutex>
#include <unordered_set>

namespace pts::rendering {

namespace {

// Slang source declaring the `[DynamicBuffer]` user attribute. Registered
// once per IGlobalSession via `addBuiltins` so that shaders can annotate
// uniform buffers for dynamic-offset dispatch without having to `import` a
// dedicated module. The attribute is read back during metadata emission (see
// `slangMetadata.cpp::has_dynamic_buffer_attr`).
constexpr const char* k_pts_attrs_builtins =
    "[__AttributeUsage(_AttributeTargets.Var)]\n"
    "public struct DynamicBufferAttribute {}\n"
    "[__AttributeUsage(_AttributeTargets.Var)]\n"
    "public struct NonFilteringAttribute {}\n";

void ensure_pts_attrs_registered(slang::IGlobalSession* gs) {
    static std::mutex s_mutex;
    static std::unordered_set<slang::IGlobalSession*> s_registered;
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_registered.insert(gs).second) return;
    gs->addBuiltins("pts_attrs.slang", k_pts_attrs_builtins);
}

}  // namespace

SlangCompileOutput run_slang(slang::IGlobalSession* global_session,
                             boost::span<const std::filesystem::path> search_paths,
                             const std::filesystem::path& slang_source,
                             const std::vector<std::string>& entry_points,
                             boost::span<const std::string_view> defines,
                             std::string_view metadata_namespace) {
    SlangCompileOutput out;

    ensure_pts_attrs_registered(global_session);

    slang::SessionDesc session_desc = {};
    slang::TargetDesc target_desc = {};
    target_desc.format = SLANG_WGSL;
    session_desc.targets = &target_desc;
    session_desc.targetCount = 1;
    // Match CLI slangc default: column-major matrix layout
    session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;

    // Source file's own directory first (matches slangc CLI default), then
    // every configured search path in order.
    std::vector<std::string> search_storage;
    search_storage.reserve(search_paths.size() + 1);
    search_storage.push_back(slang_source.parent_path().string());
    for (const auto& sp : search_paths) {
        search_storage.push_back(sp.string());
    }
    std::vector<const char*> search_ptrs;
    search_ptrs.reserve(search_storage.size());
    for (const auto& s : search_storage) {
        search_ptrs.push_back(s.c_str());
    }
    session_desc.searchPaths = search_ptrs.data();
    session_desc.searchPathCount = static_cast<SlangInt>(search_ptrs.size());

    std::vector<std::string> define_storage(defines.begin(), defines.end());
    std::vector<slang::PreprocessorMacroDesc> macros;
    macros.reserve(defines.size());
    for (const auto& d : define_storage) {
        macros.push_back({d.c_str(), "1"});
    }
    session_desc.preprocessorMacros = macros.data();
    session_desc.preprocessorMacroCount = static_cast<SlangInt>(macros.size());

    Slang::ComPtr<slang::ISession> session;
    auto hr = global_session->createSession(session_desc, session.writeRef());
    if (SLANG_FAILED(hr) || !session) {
        out.diagnostics = "Failed to create Slang session";
        return out;
    }

    auto module_name = slang_source.stem().string();
    Slang::ComPtr<slang::IBlob> diagnostics;
    auto* module = session->loadModule(module_name.c_str(), diagnostics.writeRef());
    if (diagnostics) {
        out.diagnostics = static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (!module) {
        return out;
    }

    auto dep_count = module->getDependencyFileCount();
    for (SlangInt32 i = 0; i < dep_count; ++i) {
        auto* dep_path = module->getDependencyFilePath(i);
        if (dep_path) {
            out.dependencies.emplace_back(dep_path);
        }
    }

    std::vector<Slang::ComPtr<slang::IEntryPoint>> ep_objects;
    if (entry_points.empty()) {
        // Match slangc CLI default: pick up every entry point the module declares.
        SlangInt32 defined_count = module->getDefinedEntryPointCount();
        for (SlangInt32 i = 0; i < defined_count; ++i) {
            Slang::ComPtr<slang::IEntryPoint> ep;
            hr = module->getDefinedEntryPoint(i, ep.writeRef());
            if (SLANG_FAILED(hr) || !ep) return out;
            ep_objects.push_back(std::move(ep));
        }
    } else {
        for (const auto& ep_name : entry_points) {
            SlangStage stage = SLANG_STAGE_NONE;
            if (ep_name.find("vs_") == 0 || ep_name.find("vert") == 0) {
                stage = SLANG_STAGE_VERTEX;
            } else if (ep_name.find("fs_") == 0 || ep_name.find("frag") == 0) {
                stage = SLANG_STAGE_FRAGMENT;
            } else if (ep_name.find("cs_") == 0 || ep_name.find("comp") == 0) {
                stage = SLANG_STAGE_COMPUTE;
            }

            Slang::ComPtr<slang::IEntryPoint> ep;
            hr = module->findAndCheckEntryPoint(ep_name.c_str(), stage, ep.writeRef(),
                                                diagnostics.writeRef());
            if (diagnostics) {
                out.diagnostics += static_cast<const char*>(diagnostics->getBufferPointer());
            }
            if (SLANG_FAILED(hr) || !ep) {
                return out;
            }
            ep_objects.push_back(std::move(ep));
        }
    }

    std::vector<slang::IComponentType*> components;
    components.push_back(module);
    for (auto& ep : ep_objects) components.push_back(ep.get());

    Slang::ComPtr<slang::IComponentType> program;
    hr = session->createCompositeComponentType(components.data(), components.size(),
                                               program.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        out.diagnostics += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !program) return out;

    Slang::ComPtr<slang::IComponentType> linked;
    hr = program->link(linked.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        out.diagnostics += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !linked) return out;

    Slang::ComPtr<slang::IBlob> code;
    hr = linked->getTargetCode(0, code.writeRef(), diagnostics.writeRef());
    if (diagnostics) {
        out.diagnostics += static_cast<const char*>(diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(hr) || !code) return out;

    out.wgsl.assign(static_cast<const char*>(code->getBufferPointer()), code->getBufferSize());

    if (!metadata_namespace.empty()) {
        auto* layout = linked->getLayout(0, diagnostics.writeRef());
        if (diagnostics) {
            out.diagnostics += static_cast<const char*>(diagnostics->getBufferPointer());
        }
        if (layout) {
            out.metadata_header =
                run_slang_metadata_header(global_session, layout, linked.get(), metadata_namespace,
                                          /*target_index=*/0);
        }
    }

    out.success = true;
    return out;
}

}  // namespace pts::rendering

#endif  // !__EMSCRIPTEN__
