from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, save
from conan.tools.scm import Git
import os

# Pinned commit for reproducible builds (compatible with imgui 1.92+)
# From: https://github.com/CedricGuillemet/ImGuizmo
IMGUIZMO_COMMIT = "a15acd87a3f3241a29ea1363ceafc680dca3a96b"


class ImGuizmoConan(ConanFile):
    name = "imguizmo"
    version = "1.92"
    license = "MIT"
    description = "Immediate mode 3D gizmo for scene editing and other controls based on Dear Imgui"
    homepage = "https://github.com/CedricGuillemet/ImGuizmo"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def requirements(self):
        # Depend on imgui docking from Conan
        self.requires("imgui/1.92.0-docking", transitive_headers=True)

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self, src_folder="src")

    def source(self):
        # Clone and checkout specific commit for reproducibility
        git = Git(self)
        git.clone("https://github.com/CedricGuillemet/ImGuizmo.git", target=".")
        git.checkout(IMGUIZMO_COMMIT)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()
        cmake_content = """
cmake_minimum_required(VERSION 3.15)
project(imguizmo CXX)

find_package(imgui REQUIRED)

add_library(imguizmo
    GraphEditor.cpp
    ImCurveEdit.cpp
    ImGradient.cpp
    ImGuizmo.cpp
    ImSequencer.cpp
)

target_compile_definitions(imguizmo PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

target_include_directories(imguizmo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(imguizmo PUBLIC imgui::imgui)

set_target_properties(imguizmo PROPERTIES
    PUBLIC_HEADER "GraphEditor.h;ImCurveEdit.h;ImGradient.h;ImGuizmo.h;ImSequencer.h;ImZoomSlider.h"
)

include(GNUInstallDirs)
install(TARGETS imguizmo
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
"""
        save(self, os.path.join(self.source_folder, "CMakeLists.txt"), cmake_content)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )

    def package_info(self):
        self.cpp_info.libs = ["imguizmo"]
        self.cpp_info.defines = ["IMGUI_DEFINE_MATH_OPERATORS"]
        self.cpp_info.set_property("cmake_file_name", "imguizmo")
        self.cpp_info.set_property("cmake_target_name", "imguizmo::imguizmo")
