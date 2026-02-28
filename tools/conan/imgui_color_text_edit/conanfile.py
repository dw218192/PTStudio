from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, save
from conan.tools.scm import Git
import os

# Pinned commit for reproducible builds (compatible with imgui 1.91+)
# From: https://github.com/BalazsJako/ImGuiColorTextEdit
IMGUI_COLOR_TEXT_EDIT_COMMIT = "ca2f9f1462e3b60e56351bc466acda448c5ea50d"


class ImGuiColorTextEditConan(ConanFile):
    name = "imgui_color_text_edit"
    version = "1.0"
    license = "MIT"
    description = "Syntax highlighting text editor for ImGui"
    homepage = "https://github.com/BalazsJako/ImGuiColorTextEdit"
    settings = "os", "arch", "compiler", "build_type"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def requirements(self):
        # Depend on imgui docking from Conan
        self.requires("imgui/1.92.5-docking", transitive_headers=True)

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
        git.clone("https://github.com/BalazsJako/ImGuiColorTextEdit.git", target=".")
        git.checkout(IMGUI_COLOR_TEXT_EDIT_COMMIT)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

        # Create a minimal CMakeLists.txt for building the library
        cmake_content = """
cmake_minimum_required(VERSION 3.15)
project(imgui_color_text_edit CXX)

find_package(imgui REQUIRED)

add_library(imgui_color_text_edit
    TextEditor.cpp
)

target_compile_definitions(imgui_color_text_edit PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

target_include_directories(imgui_color_text_edit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(imgui_color_text_edit PUBLIC imgui::imgui)

set_target_properties(imgui_color_text_edit PROPERTIES
    PUBLIC_HEADER "TextEditor.h"
)

include(GNUInstallDirs)
install(TARGETS imgui_color_text_edit
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
        self.cpp_info.libs = ["imgui_color_text_edit"]
        self.cpp_info.defines = ["IMGUI_DEFINE_MATH_OPERATORS"]
        self.cpp_info.set_property("cmake_file_name", "imgui_color_text_edit")
        self.cpp_info.set_property(
            "cmake_target_name", "imgui_color_text_edit::imgui_color_text_edit"
        )
