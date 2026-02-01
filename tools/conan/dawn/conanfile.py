from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.scm import Git
from pathlib import Path
import sys


class DawnConan(ConanFile):
    name = "dawn"
    version = "20250428.160623"
    license = "Apache-2.0"
    description = "Dawn WebGPU implementation"
    homepage = "https://dawn.googlesource.com/dawn"
    settings = "os", "arch", "compiler", "build_type"
    options = {}
    default_options = {}

    def layout(self) -> None:
        cmake_layout(self)

    def source(self) -> None:
        git = Git(self)
        git.clone("https://github.com/google/dawn.git", target=".")
        git.checkout(f"v{self.version}")

    def generate(self) -> None:
        tc = CMakeToolchain(self)
        tc.cache_variables["DAWN_ENABLE_INSTALL"] = True
        tc.cache_variables["DAWN_BUILD_SAMPLES"] = False
        tc.cache_variables["DAWN_BUILD_TESTS"] = False
        tc.cache_variables["DAWN_FETCH_DEPENDENCIES"] = True
        tc.cache_variables["TINT_BUILD_TESTS"] = False
        tc.cache_variables["DAWN_BUILD_MONOLITHIC_LIBRARY"] = True
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self) -> None:
        self.run(
            f"{sys.executable} tools/fetch_dawn_dependencies.py", cwd=self.source_folder
        )
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self) -> None:
        cmake = CMake(self)
        cmake.install()

    def package_info(self) -> None:
        self.cpp_info.set_property("cmake_file_name", "Dawn")
        self.cpp_info.set_property("cmake_target_name", "dawn::webgpu_dawn")
        self.cpp_info.libs = ["webgpu_dawn"]
