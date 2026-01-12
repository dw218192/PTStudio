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
    }
    default_options = {
        "pathtracer": False,
        "build_tests": True,
        "boost/*:without_atomic": False,
        "boost/*:without_chrono": True,
        "boost/*:without_container": False,
        "boost/*:without_context": False,
        "boost/*:without_contract": True,
        "boost/*:without_coroutine": True,
        "boost/*:without_date_time": True,
        "boost/*:without_exception": True,
        "boost/*:without_fiber": True,
        "boost/*:without_filesystem": False,
        "boost/*:without_graph": True,
        "boost/*:without_graph_parallel": True,
        "boost/*:without_iostreams": True,
        "boost/*:without_locale": True,
        "boost/*:without_log": True,
        "boost/*:without_math": True,
        "boost/*:without_mpi": True,
        "boost/*:without_nowide": True,
        "boost/*:without_program_options": True,
        "boost/*:without_python": True,
        "boost/*:without_random": True,
        "boost/*:without_regex": True,
        "boost/*:without_serialization": True,
        "boost/*:without_stacktrace": True,
        "boost/*:without_system": True,
        "boost/*:without_test": True,
        "boost/*:without_thread": True,
        "boost/*:without_timer": True,
        "boost/*:without_type_erasure": True,
        "boost/*:without_wave": True,
        "boost/*:without_describe": False,
        "boost/*:without_dll": False,
    }
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
        self.requires("tinyobjloader/2.0.0-rc10")
        self.requires("tcb-span/[>=0]")
        self.requires("tl-expected/[>=0]")
        self.requires("doctest/[>=0]")
        self.requires("boost/[>=0]")

        # Vulkan support
        self.requires("vulkan-headers/[>=0]")
        self.requires("vulkan-loader/[>=0]")
        self.requires("shaderc/[>=0]")

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

        tc.generate()
