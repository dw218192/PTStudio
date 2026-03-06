from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
import os


class SlangReflectConan(ConanFile):
    name = "slang_reflect"
    version = "1.0"
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"
    exports_sources = "src/*", "CMakeLists.txt"
    requires = ("slang/2026.1",)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)

        # application package_type may not get CMakeDeps find-scripts for
        # requires, so pass slang paths explicitly via cache variables.
        slang_info = self.dependencies["slang"].cpp_info
        tc.cache_variables["SLANG_INCLUDE_DIR"] = slang_info.includedirs[0]
        tc.cache_variables["SLANG_LIB_DIR"] = slang_info.libdirs[0]
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.bindirs = ["bin"]
