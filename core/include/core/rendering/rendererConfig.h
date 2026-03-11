#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace pts::rendering {

class IScenePass;

using PassFactory = std::function<std::unique_ptr<IScenePass>()>;

struct RendererConfig {
    std::string_view name;
    std::vector<PassFactory> pass_factories;
};

}  // namespace pts::rendering
