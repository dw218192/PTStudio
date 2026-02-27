from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import cross_building
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.env import VirtualBuildEnv
from conan.tools.files import copy, get, load, rm, rmdir
from conan.tools.gnu import PkgConfigDeps
import os
import re


class OneTBBConan(ConanFile):
    name = "onetbb"
    version = "2021.12.0"
    license = "Apache-2.0"
    description = "oneAPI Threading Building Blocks (oneTBB)"
    homepage = "https://github.com/oneapi-src/oneTBB"
    url = "https://github.com/oneapi-src/oneTBB"
    topics = ("tbb", "threading", "parallelism", "tbbmalloc")

    # "library" (not "shared-library") so the `shared` option is respected.
    # The Conan Center recipe hard-codes shared-library with no shared option,
    # which makes static Emscripten builds impossible.
    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "tbbmalloc": [True, False],
        "tbbproxy": [True, False],
        "tbbbind": [True, False],
        "interprocedural_optimization": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "tbbmalloc": True,
        "tbbproxy": True,
        "tbbbind": True,
        "interprocedural_optimization": True,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
        if self.settings.os == "Emscripten":
            del self.options.tbbbind
            del self.options.tbbproxy
            del self.options.interprocedural_optimization

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")
        if not self.options.tbbmalloc:
            self.options.rm_safe("tbbproxy")

    def layout(self):
        cmake_layout(self, src_folder="src")

    def requirements(self):
        if self.options.get_safe("tbbbind", False):
            self.requires("hwloc/2.12.2")

    def validate(self):
        if "hwloc" in self.dependencies.direct_host:
            if self.dependencies["hwloc"].package_type != "shared-library":
                raise ConanInvalidConfiguration(
                    f"{self.ref} requires hwloc/*:shared=True"
                )

    def build_requirements(self):
        if self.options.get_safe("tbbbind", False) and not cross_building(self):
            if not self.conf.get("tools.gnu:pkg_config", check_type=str):
                self.tool_requires("pkgconf/2.1.0")

    def source(self):
        get(
            self,
            url=f"https://github.com/oneapi-src/oneTBB/archive/v{self.version}.tar.gz",
            strip_root=True,
        )

    def generate(self):
        env = VirtualBuildEnv(self)
        env.generate()

        tc = CMakeToolchain(self)
        tc.variables["TBB_TEST"] = False
        tc.variables["TBB_STRICT"] = False
        tc.variables["TBBMALLOC_BUILD"] = self.options.tbbmalloc
        if self.options.get_safe("interprocedural_optimization") is not None:
            tc.variables["TBB_ENABLE_IPO"] = self.options.interprocedural_optimization
        if self.options.get_safe("tbbmalloc"):
            tc.variables["TBBMALLOC_PROXY_BUILD"] = self.options.get_safe(
                "tbbproxy", False
            )
        tc.variables["TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH"] = not self.options.get_safe(
            "tbbbind", False
        )
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared

        # Cross-compilation: pass hwloc paths directly to CMake
        if self.options.get_safe("tbbbind", False) and cross_building(self):
            hwloc_pkg = self.dependencies["hwloc"].package_folder
            hwloc_lib_name = (
                "hwloc.lib"
                if self.settings.os == "Windows"
                else "libhwloc.dylib"
                if self.settings.os == "Macos"
                else "libhwloc.so"
            )
            tc.variables["CMAKE_HWLOC_2_5_LIBRARY_PATH"] = os.path.join(
                hwloc_pkg, "lib", hwloc_lib_name
            ).replace("\\", "/")
            tc.variables["CMAKE_HWLOC_2_5_INCLUDE_PATH"] = os.path.join(
                hwloc_pkg, "include"
            ).replace("\\", "/")
            if self.settings.os == "Windows":
                tc.variables["CMAKE_HWLOC_2_5_DLL_PATH"] = os.path.join(
                    hwloc_pkg, "bin", "hwloc.dll"
                ).replace("\\", "/")

        tc.generate()

        if "hwloc" in self.dependencies.direct_host:
            deps = PkgConfigDeps(self)
            deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE.txt",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "share"))
        rm(self, "*.pdb", os.path.join(self.package_folder, "bin"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "TBB")
        self.cpp_info.set_property("pkg_config_name", "tbb")
        self.cpp_info.set_property(
            "cmake_config_version_compat", "AnyNewerVersion"
        )

        def lib_name(name):
            if self.settings.build_type == "Debug":
                return name + "_debug"
            return name

        # libtbb
        tbb = self.cpp_info.components["libtbb"]
        tbb.set_property("cmake_target_name", "TBB::tbb")
        tbb.libs = [lib_name("tbb")]
        if self.settings.os == "Windows":
            version_info = load(
                self,
                os.path.join(
                    self.package_folder,
                    "include",
                    "oneapi",
                    "tbb",
                    "version.h",
                ),
            )
            binary_version = re.sub(
                r".*" + re.escape("#define __TBB_BINARY_VERSION ") + r"(\d+).*",
                r"\1",
                version_info,
                flags=re.MULTILINE | re.DOTALL,
            )
            tbb.libs.append(lib_name(f"tbb{binary_version}"))
        if self.settings.os in ["Linux", "FreeBSD"]:
            tbb.system_libs = ["m", "dl", "rt", "pthread"]

        # tbbmalloc
        if self.options.tbbmalloc:
            tbbmalloc = self.cpp_info.components["tbbmalloc"]
            tbbmalloc.set_property("cmake_target_name", "TBB::tbbmalloc")
            tbbmalloc.libs = [lib_name("tbbmalloc")]
            if self.settings.os in ["Linux", "FreeBSD"]:
                tbbmalloc.system_libs = ["dl", "pthread"]

            # tbbmalloc_proxy
            if self.options.get_safe("tbbproxy"):
                tbbproxy = self.cpp_info.components["tbbmalloc_proxy"]
                tbbproxy.set_property(
                    "cmake_target_name", "TBB::tbbmalloc_proxy"
                )
                tbbproxy.libs = [lib_name("tbbmalloc_proxy")]
                tbbproxy.requires = ["tbbmalloc"]
                if self.settings.os in ["Linux", "FreeBSD"]:
                    tbbproxy.system_libs = ["m", "dl", "pthread"]
