from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


class UsdzPackConan(ConanFile):
    name = "usdz_pack"
    version = "1.0"
    description = "CLI tool wrapping UsdUtilsCreateNewUsdzPackage"
    package_type = "application"
    settings = "os", "arch", "compiler", "build_type"
    exports_sources = "CMakeLists.txt", "usdzPack.cpp"

    def requirements(self):
        self.requires("openusd/25.11-dev")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.bindirs = ["bin"]
