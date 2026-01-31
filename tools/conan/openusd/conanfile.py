from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get
from conan.tools.scm import Version
import os

required_conan_version = ">=1.53.0"


class OpenUSDConan(ConanFile):
    name = "openusd"
    version = "25.02"
    license = "Apache-2.0"
    description = "Universal Scene Description (USD) - minimal core build"
    homepage = "https://openusd.org/"
    url = "https://github.com/PixarAnimationStudios/OpenUSD"
    topics = ("3d", "scene", "usd", "pixar")
    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        # onetbb requires hwloc to be shared
        "onetbb/*:hwloc": True,
        "hwloc/*:shared": True,
    }

    short_paths = True  # Important for Windows - USD has deep paths

    @property
    def _min_cppstd(self):
        return 17

    @property
    def _compilers_minimum_version(self):
        return {
            "apple-clang": "13",
            "clang": "7",
            "gcc": "9",
            "msvc": "191",
            "Visual Studio": "15",
        }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self) -> None:
        cmake_layout(self, src_folder="src")

    def requirements(self) -> None:
        self.requires("onetbb/2021.12.0", transitive_headers=True, transitive_libs=True)

    def build_requirements(self) -> None:
        self.tool_requires("cmake/[>=3.24 <4]")

    def validate(self):
        if self.settings.compiler.cppstd:
            check_min_cppstd(self, self._min_cppstd)
        minimum_version = self._compilers_minimum_version.get(
            str(self.settings.compiler), False
        )
        if (
            minimum_version
            and Version(self.settings.compiler.version) < minimum_version
        ):
            raise ConanInvalidConfiguration(
                f"{self.ref} requires C++{self._min_cppstd}, which your compiler does not support."
            )
        # onetbb forbids static builds
        if not self.options.shared:
            raise ConanInvalidConfiguration(
                "openusd does not support static build because onetbb recipe forbids it"
            )

    def source(self) -> None:
        get(
            self,
            f"https://github.com/PixarAnimationStudios/OpenUSD/archive/refs/tags/v{self.version}.tar.gz",
            strip_root=True,
        )

    def generate(self) -> None:
        tc = CMakeToolchain(self)
        # Minimal build configuration
        tc.variables["PXR_BUILD_USDVIEW"] = False
        tc.variables["PXR_BUILD_TESTS"] = False
        tc.variables["PXR_BUILD_EXAMPLES"] = False
        tc.variables["PXR_BUILD_TUTORIALS"] = False
        tc.variables["PXR_BUILD_HTML_DOCUMENTATION"] = False
        tc.variables["PXR_ENABLE_PYTHON_SUPPORT"] = False
        # Disable imaging and optional features
        tc.variables["PXR_BUILD_IMAGING"] = False
        tc.variables["PXR_BUILD_USD_IMAGING"] = False
        tc.variables["PXR_ENABLE_PTEX_SUPPORT"] = False
        tc.variables["PXR_ENABLE_OPENVDB_SUPPORT"] = False
        tc.variables["PXR_ENABLE_MATERIALX_SUPPORT"] = False
        # Build options
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.variables["PXR_BUILD_MONOLITHIC"] = False
        # Tell USD to use Conan's TBB target
        tc.variables["TBB_tbb_LIBRARY"] = "onetbb::onetbb"
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self) -> None:
        copy(
            self,
            "LICENSE.txt",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()

    def package_info(self) -> None:
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["m", "pthread", "dl"])

        # USD core components with proper TBB dependencies (from reference)
        self.cpp_info.components["usd_arch"].libs = ["usd_arch"]

        self.cpp_info.components["usd_tf"].libs = ["usd_tf"]
        self.cpp_info.components["usd_tf"].requires = ["usd_arch", "onetbb::libtbb"]

        self.cpp_info.components["usd_gf"].libs = ["usd_gf"]
        self.cpp_info.components["usd_gf"].requires = ["usd_arch", "usd_tf"]

        self.cpp_info.components["usd_js"].libs = ["usd_js"]
        self.cpp_info.components["usd_js"].requires = ["usd_tf"]

        self.cpp_info.components["usd_trace"].libs = ["usd_trace"]
        self.cpp_info.components["usd_trace"].requires = [
            "usd_arch",
            "usd_tf",
            "usd_js",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_work"].libs = ["usd_work"]
        self.cpp_info.components["usd_work"].requires = [
            "usd_tf",
            "usd_trace",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_plug"].libs = ["usd_plug"]
        self.cpp_info.components["usd_plug"].requires = [
            "usd_arch",
            "usd_tf",
            "usd_js",
            "usd_trace",
            "usd_work",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_vt"].libs = ["usd_vt"]
        self.cpp_info.components["usd_vt"].requires = [
            "usd_arch",
            "usd_tf",
            "usd_gf",
            "usd_trace",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_ts"].libs = ["usd_ts"]
        self.cpp_info.components["usd_ts"].requires = [
            "usd_arch",
            "usd_gf",
            "usd_plug",
            "usd_tf",
            "usd_trace",
            "usd_vt",
        ]

        self.cpp_info.components["usd_ar"].libs = ["usd_ar"]
        self.cpp_info.components["usd_ar"].requires = [
            "usd_arch",
            "usd_js",
            "usd_tf",
            "usd_plug",
            "usd_vt",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_kind"].libs = ["usd_kind"]
        self.cpp_info.components["usd_kind"].requires = ["usd_tf", "usd_plug"]

        self.cpp_info.components["usd_sdf"].libs = ["usd_sdf"]
        self.cpp_info.components["usd_sdf"].requires = [
            "usd_arch",
            "usd_tf",
            "usd_ts",
            "usd_gf",
            "usd_trace",
            "usd_vt",
            "usd_work",
            "usd_ar",
        ]

        self.cpp_info.components["usd_pcp"].libs = ["usd_pcp"]
        self.cpp_info.components["usd_pcp"].requires = [
            "usd_tf",
            "usd_trace",
            "usd_vt",
            "usd_sdf",
            "usd_work",
            "usd_ar",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_usd"].libs = ["usd_usd"]
        self.cpp_info.components["usd_usd"].requires = [
            "usd_arch",
            "usd_kind",
            "usd_pcp",
            "usd_sdf",
            "usd_ar",
            "usd_plug",
            "usd_tf",
            "usd_trace",
            "usd_vt",
            "usd_work",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_usdGeom"].libs = ["usd_usdGeom"]
        self.cpp_info.components["usd_usdGeom"].requires = [
            "usd_js",
            "usd_tf",
            "usd_plug",
            "usd_vt",
            "usd_sdf",
            "usd_trace",
            "usd_usd",
            "usd_work",
            "onetbb::libtbb",
        ]

        self.cpp_info.components["usd_sdr"].libs = ["usd_sdr"]
        self.cpp_info.components["usd_sdr"].requires = [
            "usd_tf",
            "usd_vt",
            "usd_ar",
            "usd_sdf",
        ]

        self.cpp_info.components["usd_usdShade"].libs = ["usd_usdShade"]
        self.cpp_info.components["usd_usdShade"].requires = [
            "usd_tf",
            "usd_vt",
            "usd_js",
            "usd_sdf",
            "usd_sdr",
            "usd_usd",
            "usd_usdGeom",
        ]

        self.cpp_info.components["usd_usdLux"].libs = ["usd_usdLux"]
        self.cpp_info.components["usd_usdLux"].requires = [
            "usd_tf",
            "usd_vt",
            "usd_sdf",
            "usd_usd",
            "usd_usdGeom",
            "usd_usdShade",
        ]

        self.cpp_info.components["usd_usdUtils"].libs = ["usd_usdUtils"]
        self.cpp_info.components["usd_usdUtils"].requires = [
            "usd_arch",
            "usd_tf",
            "usd_gf",
            "usd_sdf",
            "usd_usd",
            "usd_usdGeom",
            "usd_usdShade",
        ]
