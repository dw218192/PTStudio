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
        "build_tests": [True, False],
        "windowing": ["glfw", "null"],
    }
    default_options = {
        "build_tests": True,
        "windowing": "glfw",
        # Boost configuration - need filesystem for path operations
        "boost/*:without_filesystem": False,
        # Tracy: manual lifetime prevents static destructor deadlock at exit.
        # on_demand: profiler thread uses select() with timeout, not blocking accept().
        "tracy/*:on_demand": True,
        "tracy/*:delayed_init": True,
        "tracy/*:manual_lifetime": True,
    }

    def requirements(self):
        # Core dependencies, strict compatibility
        self.requires("fmt/10.2.1")
        self.requires("spdlog/1.14.1")
        self.requires("nlohmann_json/3.12.0")

        # Graphics libraries (GLFW not needed for Emscripten - use -sUSE_GLFW=3)
        if self.options.windowing == "glfw" and self.settings.os != "Emscripten":
            self.requires("glfw/[>=0]")
        self.requires("glm/[>=0]")

        # Utility libraries
        self.requires("doctest/[>=0]")
        self.requires("boost/[>=0]")
        self.requires("cxxopts/[>=3]")
        # Scene description
        self.requires("openusd/25.11-dev")

        # These dependencies don't work or aren't needed for Emscripten
        if self.settings.os != "Emscripten":
            # WebGPU backend (Emscripten gets Dawn via emdawnwebgpu emcc port)
            self.requires("dawn/20251002.162335")
            # File dialogs (not applicable in browser)
            self.requires("portable-file-dialogs/0.1.0")
            # Profiler
            self.requires("tracy/0.13.1")
            # Slang compiler library for in-process shader compilation
            self.requires("slang/2026.5.2")

        # GUI libraries (from Conan)
        self.requires("imgui/1.92.5-docking")
        self.requires("imguizmo/1.92")
        self.requires("stb/[>=0]")
        # OpenEXR <3.4: 3.4+ adds openjph (JPEG2000) which drags in libtiff,
        # libjpeg, libdeflate, xz_utils — unnecessary deps that also break
        # Emscripten cross-compile and invalidate Conan binary caches on CI.
        self.requires("openexr/[>=3.1 <3.4]")

    def build_requirements(self):
        if self.settings.os == "Emscripten":
            self.tool_requires("emsdk/4.0.10")
            self.tool_requires("ninja/1.13.2")

    def configure(self):
        # Configure package options
        if self.options.get_safe("shared"):
            del self.options.shared
        
        # Disable Boost stacktrace features that don't work on Emscripten
        if self.settings.os == "Emscripten":
            # Use header-only mode to avoid all compilation issues on Emscripten
            self.options["boost"].header_only = True
            self.options["boost"].without_stacktrace = True
            self.options["boost"].without_locale = True
            self.options["boost"].without_log = True
            self.options["boost"].without_context = True
            self.options["boost"].without_coroutine = True
            self.options["boost"].without_fiber = True
            self.options["boost"].without_test = True
            self.options["boost"].without_type_erasure = True
            self.options["boost"].without_process = True
            self.options["boost"].without_thread = True
            self.options["boost"].without_filesystem = True
            self.options["boost"].without_program_options = True
            self.options["boost"].without_regex = True
            self.options["boost"].without_math = True
            self.options["boost"].without_random = True
            self.options["boost"].without_serialization = True
            self.options["boost"].without_wave = True
            self.options["boost"].without_iostreams = True
            self.options["boost"].without_graph = True
            self.options["boost"].without_timer = True
            self.options["boost"].without_url = True
            self.options["boost"].without_nowide = True
            self.options["boost"].without_contract = True
            self.options["boost"].without_json = True
            self.options["boost"].without_charconv = True
            self.options["boost"].without_chrono = True
            self.options["boost"].without_atomic = True
            self.options["boost"].without_date_time = True
            self.options["boost"].without_exception = True
            self.options["boost"].without_container = True
            self.options["bzip2"].build_executable = False
            # Disable hwloc - uses autotools which doesn't work on Windows for cross-compile
            self.options["onetbb"].tbbbind = False
            # Force static builds for WASM
            self.options["openusd"].shared = False
            # onetbb recipe is strictly shared-library, so we cannot force static via options

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
        tc.cache_variables["CORE_BUILD_TESTS"] = self.options.build_tests
        # Ensure spdlog uses external fmt library
        tc.cache_variables["SPDLOG_FMT_EXTERNAL"] = "ON"

        tc.cache_variables["PTS_WINDOWING"] = self.options.windowing
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()
