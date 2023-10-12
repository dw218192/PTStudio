#pragma once

#define GLM_FORCE_SILENT_WARNINGS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <tiny_obj_loader.h>
#include <stb_image_write.h>

#include <imgui.h>
#ifdef _WIN32
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_win32.h>
#elif defined(__linux__)
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#else
#error "Unsupported platform"
#endif

// c++23 expected
#include <tl/expected.hpp>

inline tl::unexpected<char const*> unexpected_gl_error(GLenum err) {
	return tl::unexpected{ reinterpret_cast<char const*>(glewGetErrorString(err)) };
}