#pragma once

struct RenderConfig {
	RenderConfig(unsigned width, unsigned height, float fovy, float max_fps)
		: width{ width }, height{ height }, fovy{ fovy }, max_fps{ max_fps }, min_frame_time{ 1.0f / max_fps }
	{ }

	bool operator==(RenderConfig const& other) const noexcept {
		return width == other.width &&
			height == other.height &&
			std::abs(fovy - other.fovy) < 1e-6f &&
			std::abs(max_fps - other.max_fps) < 1e-6f;
	}

	bool operator!=(RenderConfig const& other) const noexcept {
		return !(*this == other);
	}

	bool is_valid() const noexcept {
		return width != 0 && height != 0 && fovy >= 20 && max_fps >= 30;
	}

	unsigned width, height;
	float fovy;
	float max_fps;
	float min_frame_time;
};