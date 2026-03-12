#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pts::rendering {

class IScenePass;

using PassFactory = std::function<std::unique_ptr<IScenePass>()>;

struct RendererConfig {
    std::string name;
    std::vector<PassFactory> pass_factories;
};

}  // namespace pts::rendering
