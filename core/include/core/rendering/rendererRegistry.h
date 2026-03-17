#pragma once

#include <core/diagnostics.h>
#include <core/rendering/rendererConfig.h>

#include <string>
#include <string_view>
#include <vector>

namespace pts::rendering {

struct RendererEntry {
    std::string name;
    PassFactory factory;
};

class RendererRegistry {
   public:
    static auto& entries() {
        static std::vector<RendererEntry> e;
        return e;
    }

    static int add(RendererEntry entry) {
        entries().push_back(std::move(entry));
        return 0;
    }

    static PassFactory find(std::string_view name) {
        for (auto& e : entries()) {
            if (e.name == name) return e.factory;
        }
        PANIC("RendererRegistry::find: no renderer registered with the requested name");
    }
};

}  // namespace pts::rendering

#define REGISTER_RENDERER(name, PassClass)                                       \
    static int s_register_##PassClass = ::pts::rendering::RendererRegistry::add( \
        {name, [] { return std::make_unique<PassClass>(); }})
