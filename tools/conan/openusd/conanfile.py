from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get, save
from conan.tools.scm import Git, Version
import os

required_conan_version = ">=1.53.0"

OPENUSD_COMMIT = "2f88bd53bd2690998c3d7507d24e8d213068deb1"


class OpenUSDConan(ConanFile):
    name = "openusd"
    version = "25.11-dev"
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
        # Emscripten doesn't support hwloc
        if self.settings.os == "Emscripten":
            self.options["onetbb"].tbbbind = False
        else:
            # onetbb requires hwloc to be shared on desktop platforms
            self.options["onetbb"].hwloc = True
            self.options["hwloc"].shared = True
        # OpenSubdiv: CPU-only, no GPU backends
        self.options["opensubdiv"].with_opengl = False
        self.options["opensubdiv"].with_tbb = False
        self.options["opensubdiv"].with_omp = False
        self.options["opensubdiv"].with_cuda = False
        self.options["opensubdiv"].with_clew = False
        self.options["opensubdiv"].with_opencl = False
        self.options["opensubdiv"].with_dx = False
        self.options["opensubdiv"].with_metal = False

    def layout(self) -> None:
        cmake_layout(self, src_folder="src")

    def requirements(self) -> None:
        self.requires("onetbb/2021.12.0", transitive_headers=True, transitive_libs=True)
        self.requires("opensubdiv/3.6.0", transitive_headers=True, transitive_libs=True)

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

    def source(self) -> None:
        git = Git(self)
        git.clone("https://github.com/PixarAnimationStudios/OpenUSD.git", target=".")
        git.checkout(OPENUSD_COMMIT)

    def generate(self) -> None:
        tc = CMakeToolchain(self)
        # Minimal build configuration
        tc.variables["PXR_BUILD_USDVIEW"] = False
        tc.variables["PXR_BUILD_TESTS"] = False
        tc.variables["PXR_BUILD_EXAMPLES"] = False
        tc.variables["PXR_BUILD_TUTORIALS"] = False
        tc.variables["PXR_BUILD_HTML_DOCUMENTATION"] = False
        tc.variables["PXR_ENABLE_PYTHON_SUPPORT"] = False
        # Enable imaging subset (pxOsd + geomUtil only, no full imaging stack)
        tc.variables["PXR_BUILD_IMAGING"] = True
        tc.variables["PXR_BUILD_USD_IMAGING"] = False
        tc.variables["PXR_ENABLE_GL_SUPPORT"] = False
        tc.variables["PXR_ENABLE_VULKAN_SUPPORT"] = False
        tc.variables["PXR_ENABLE_METAL_SUPPORT"] = False
        tc.variables["PXR_ENABLE_PTEX_SUPPORT"] = False
        tc.variables["PXR_ENABLE_OPENVDB_SUPPORT"] = False
        tc.variables["PXR_ENABLE_MATERIALX_SUPPORT"] = False
        # Disable validation for WASM - depends on usd libraries excluded on EMSCRIPTEN
        if self.settings.os == "Emscripten":
            tc.variables["PXR_BUILD_USD_VALIDATION"] = False
        # Build options
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.variables["PXR_BUILD_MONOLITHIC"] = False
        # Tell USD to use Conan's TBB target
        tc.variables["TBB_tbb_LIBRARY"] = "onetbb::onetbb"
        # Pass OpenSubdiv paths so USD's FindOpenSubdiv.cmake can locate them
        osd = self.dependencies["opensubdiv"]
        osd_root = osd.package_folder.replace("\\", "/")
        tc.cache_variables["OPENSUBDIV_INCLUDE_DIR"] = f"{osd_root}/include"
        osd_libdir = f"{osd_root}/lib"
        if self.settings.os == "Windows":
            tc.cache_variables["OPENSUBDIV_OSDCPU_LIBRARY"] = f"{osd_libdir}/osdCPU.lib"
        else:
            tc.cache_variables["OPENSUBDIV_OSDCPU_LIBRARY"] = f"{osd_libdir}/libosdCPU.a"
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self) -> None:
        # Patch imaging CMakeLists to build pxOsd, geomUtil, hf, cameraUtil, hd
        # (skip hdSt, hgi, etc.).  Order matters: hd depends on hf, cameraUtil, pxOsd.
        imaging_cmakelists = os.path.join(self.source_folder, "pxr", "imaging", "CMakeLists.txt")
        save(self, imaging_cmakelists,
             "add_subdirectory(hf)\n"
             "add_subdirectory(cameraUtil)\n"
             "add_subdirectory(pxOsd)\n"
             "add_subdirectory(geomUtil)\n"
             "add_subdirectory(hd)\n")
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

        self.cpp_info.components["usd_pxOsd"].libs = ["usd_pxOsd"]
        self.cpp_info.components["usd_pxOsd"].requires = [
            "usd_tf",
            "usd_gf",
            "usd_vt",
            "opensubdiv::opensubdiv",
        ]

        self.cpp_info.components["usd_geomUtil"].libs = ["usd_geomUtil"]
        self.cpp_info.components["usd_geomUtil"].requires = [
            "usd_arch",
            "usd_gf",
            "usd_tf",
            "usd_vt",
            "usd_pxOsd",
        ]

        self.cpp_info.components["usd_hf"].libs = ["usd_hf"]
        self.cpp_info.components["usd_hf"].requires = ["usd_tf", "usd_plug"]

        self.cpp_info.components["usd_cameraUtil"].libs = ["usd_cameraUtil"]
        self.cpp_info.components["usd_cameraUtil"].requires = ["usd_tf", "usd_gf"]

        self.cpp_info.components["usd_hd"].libs = ["usd_hd"]
        self.cpp_info.components["usd_hd"].requires = [
            "usd_arch",
            "usd_tf",
            "usd_gf",
            "usd_vt",
            "usd_sdf",
            "usd_sdr",
            "usd_trace",
            "usd_work",
            "usd_plug",
            "usd_hf",
            "usd_cameraUtil",
            "usd_pxOsd",
            "onetbb::libtbb",
        ]

        # USD installs DLLs in lib/ alongside .lib files.  With components
        # defined, VirtualRunEnv only reads component-level bindirs — set
        # them so the DLL directory appears on PATH at runtime.
        if self.settings.os == "Windows":
            for comp in self.cpp_info.components.values():
                comp.bindirs = ["bin", "lib"]
