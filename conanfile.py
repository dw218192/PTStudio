from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake
from conan.tools.files import get, copy
from conan.tools.scm import Git
import os


class PTStudioConan(ConanFile):
    name = "ptstudio"
    version = "1.0.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    def requirements(self):
        # Core dependencies, strict compatibility
        self.requires("fmt/10.2.1")
        self.requires("spdlog/1.14.1")
        self.requires("nlohmann_json/3.12.0")

        # Graphics libraries
        self.requires("glfw/[>=0]")
        self.requires("glew/[>=0]")
        self.requires("glm/[>=0]")

        # Utility libraries
        self.requires("stb/[>=0]")
        self.requires("tinyobjloader/[>=0]")

        # Vulkan support
        self.requires("vulkan-headers/[>=0]")
        self.requires("vulkan-loader/[>=0]")
        self.requires("shaderc/[>=0]")

        # Note: Some dependencies are built from source in ext/:
        # - imgui (custom build with docking branch)
        # - imguizmo
        # - ImGuiColorTextEdit
        # - expected (header-only)
        # - span (header-only)
        # - nativefiledialog
        # These will be handled in Meson build files

    def configure(self):
        # Configure package options
        if self.options.get_safe("shared"):
            del self.options.shared

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["glew-cmake_BUILD_SHARED"] = False
        tc.cache_variables["glew-cmake_BUILD_STATIC"] = True
        tc.cache_variables["EXPECTED_BUILD_TESTS"] = False
        tc.cache_variables["JSON_BuildTests"] = False

        tc.generate()
