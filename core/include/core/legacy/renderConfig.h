#pragma once
namespace PTS {
struct RenderConfig {
    RenderConfig(unsigned width, unsigned height, float fovy, float max_fps)
        : width{width},
          height{height},
          fovy{fovy},
          max_fps{max_fps},
          min_frame_time{1.0f / max_fps} {
    }

    auto operator==(RenderConfig const& other) const noexcept -> bool {
        return width == other.width && height == other.height &&
               std::abs(fovy - other.fovy) < 1e-6f && std::abs(max_fps - other.max_fps) < 1e-6f;
    }

    auto operator!=(RenderConfig const& other) const noexcept -> bool {
        return !(*this == other);
    }

    auto is_valid() const noexcept -> bool {
        return width != 0 && height != 0 && fovy >= 20 && max_fps >= 30;
    }
    auto get_aspect() const noexcept -> float {
        return width / static_cast<float>(height);
    }

    unsigned width, height;
    float fovy;
    float max_fps;
    float min_frame_time;
};
}  // namespace PTS