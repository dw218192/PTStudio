#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pts::rendering {

class IScenePass;
class ShaderLoader;

using PassFactory = std::function<std::unique_ptr<IScenePass>(const ShaderLoader&)>;

struct RendererConfig {
    std::string name;
    PassFactory factory;
};

}  // namespace pts::rendering
