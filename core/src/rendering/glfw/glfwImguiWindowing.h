#pragma once

#include <core/loggingManager.h>
#include <core/rendering/windowing.h>

#include <memory>

#include "../imguiBackend.h"

namespace spdlog {
class logger;
}

namespace pts::rendering {
class GlfwImguiWindowing final : public IImguiWindowing {
   public:
    GlfwImguiWindowing(IViewport& viewport, pts::LoggingManager& logging_manager);
    ~GlfwImguiWindowing() override;

    GlfwImguiWindowing(const GlfwImguiWindowing&) = delete;
    GlfwImguiWindowing& operator=(const GlfwImguiWindowing&) = delete;
    GlfwImguiWindowing(GlfwImguiWindowing&&) = delete;
    GlfwImguiWindowing& operator=(GlfwImguiWindowing&&) = delete;

    void new_frame() override;

   private:
    std::shared_ptr<spdlog::logger> m_logger;
    bool m_initialized{false};
};
}  // namespace pts::rendering
