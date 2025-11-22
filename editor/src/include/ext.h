#pragma once

#define GLM_FORCE_SILENT_WARNINGS
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <stb_image_write.h>
#include <tiny_obj_loader.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
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
