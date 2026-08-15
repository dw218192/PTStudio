#pragma once

#include <functional>
#include <string>
#include <vector>

namespace pts {

class LoadingOverlay {
   public:
    struct TrackedTask {
        std::string name;
        std::function<bool()> is_done;
        std::function<float()> progress;
        std::function<std::string()> status;
    };

    void track(TrackedTask task);
    bool draw();
    bool has_active_tasks() const;

   private:
    std::vector<TrackedTask> m_tasks;
};

}  // namespace pts
