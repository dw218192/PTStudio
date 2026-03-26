#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pts::rendering {

class IRenderer;
class ShaderLoader;

using RendererFactory = std::function<std::unique_ptr<IRenderer>(const ShaderLoader&)>;

struct RendererConfig {
    std::string name;
    RendererFactory factory;
};

}  // namespace pts::rendering
