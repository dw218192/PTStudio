from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake
from conan.tools.files import get, copy
from conan.tools.scm import Git
import os
import shutil


class PTStudioConan(ConanFile):
    name = "ptstudio"
    version = "1.0.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "pathtracer": [True, False],
        "build_tests": [True, False],
        "windowing": ["glfw", "null"],
    }
    default_options = {
        "pathtracer": False,
        "build_tests": True,
        "windowing": "glfw",
        # Boost configuration - need filesystem for DLL loading
        "boost/*:without_filesystem": False,
    }
    def requirements(self):
        # Core dependencies, strict compatibility
        self.requires("fmt/10.2.1")
        self.requires("spdlog/1.14.1")
        self.requires("nlohmann_json/3.12.0")

        # Graphics libraries
        if self.options.windowing == "glfw":
            self.requires("glfw/[>=0]")
        self.requires("glm/[>=0]")

        # Utility libraries
        self.requires("stb/[>=0]")
        self.requires("tinyobjloader/2.0.0-rc10")
        self.requires("tcb-span/[>=0]")
        self.requires("tl-expected/[>=0]")
        self.requires("doctest/[>=0]")
        self.requires("boost/[>=0]")
        # Shader toolchain (slangc)
        self.requires("slang/2026.1")
        # WebGPU backend
        self.requires("dawn/20250428.160623")
        # Scene description
        self.requires("openusd/25.02")

        # Note: Some dependencies are built from source in ext/:
        # - imgui (custom build with docking branch)
        # - imguizmo
        # - ImGuiColorTextEdit
        # - nativefiledialog

    def configure(self):
        # Configure package options
        if self.options.get_safe("shared"):
            del self.options.shared

    def generate(self):
        # Use Ninja generator if available
        if shutil.which("ninja") is not None:
            self.output.info("Using Ninja generator")
            tc = CMakeToolchain(self, generator="Ninja")
            tc.cache_variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        else:
            self.output.info("Ninja not found, using default CMake generator")
            tc = CMakeToolchain(self)

        # forward options to CMakeLists.txt
        tc.cache_variables["PTSTUDIO_PATHTRACER"] = self.options.pathtracer
        tc.cache_variables["CORE_BUILD_TESTS"] = self.options.build_tests
        # Ensure spdlog uses external fmt library
        tc.cache_variables["SPDLOG_FMT_EXTERNAL"] = "ON"

        tc.cache_variables["PTS_WINDOWING"] = self.options.windowing
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()
