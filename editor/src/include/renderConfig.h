#pragma once

struct RenderConfig {
	RenderConfig(unsigned width, unsigned height, float fovy, float max_fps)
		: width{ width }, height{ height }, fovy{ fovy }, max_fps{ max_fps }, min_frame_time{ 1.0f / max_fps }
	{ }

	unsigned width, height;
	float fovy;
	float max_fps;
	float min_frame_time;
};