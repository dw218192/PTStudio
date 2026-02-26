from conan import ConanFile
from conan.tools.build import cross_building
from conan.tools.env import Environment
from conan.tools.files import chdir, copy, get, replace_in_file
from conan.tools.layout import basic_layout
import json
import os


class EmSDKConan(ConanFile):
    name = "emsdk"
    version = "4.0.10"
    license = "MIT"
    description = "Emscripten SDK. Emscripten is an Open Source LLVM to JavaScript compiler"
    homepage = "https://github.com/emscripten-core/emsdk"
    topics = ("emsdk", "emscripten", "sdk")
    settings = "os", "arch", "compiler", "build_type"

    short_paths = True

    @property
    def _settings_build(self):
        return getattr(self, "settings_build", self.settings)

    def layout(self):
        basic_layout(self, src_folder="src")

    def package_id(self):
        del self.info.settings.compiler
        del self.info.settings.build_type

    def source(self):
        get(
            self,
            url=f"https://github.com/emscripten-core/emsdk/archive/{self.version}.tar.gz",
            destination=self.source_folder,
            strip_root=True,
        )

    @property
    def _relative_paths(self):
        paths = ["bin", os.path.join("bin", "upstream", "emscripten")]
        # emsdk installs Node.js into bin/node/<version>/bin/
        node_root = os.path.join(self.package_folder, "bin", "node")
        if os.path.isdir(node_root):
            for entry in os.listdir(node_root):
                node_bin = os.path.join("bin", "node", entry, "bin")
                if os.path.isdir(os.path.join(self.package_folder, node_bin)):
                    paths.append(node_bin)
        return paths

    @property
    def _paths(self):
        return [os.path.join(self.package_folder, path) for path in self._relative_paths]

    @property
    def _emsdk(self):
        return os.path.join(self.package_folder, "bin")

    @property
    def _emscripten(self):
        return os.path.join(self.package_folder, "bin", "upstream", "emscripten")

    @property
    def _em_config(self):
        return os.path.join(self.package_folder, "bin", ".emscripten")

    @property
    def _em_cache(self):
        # Place cache inside EMSCRIPTEN_ROOT so emscripten finds it by default
        # without EM_CACHE env var. Conan's full_deploy deployer rewrites
        # buildenv_info paths to the deploy folder, but CMake's compiler path
        # (from conf_info) stays in the original Conan package. On Windows CI
        # these can be on different drives, and emscripten's os.path.relpath()
        # between EM_CACHE and EMSCRIPTEN_ROOT crashes with ValueError.
        return os.path.join(self._emscripten, "cache")

    def generate(self):
        env = Environment()
        env.prepend_path("PATH", self._paths)
        env.define_path("EMSDK", self._emsdk)
        env.define_path("EMSCRIPTEN", self._emscripten)
        env.define_path("EM_CONFIG", self._em_config)
        env.define_path("EM_CACHE", self._em_cache)
        env.vars(self, scope="emsdk").save_script("emsdk_env_file")

    @staticmethod
    def _chmod_plus_x(filename):
        if os.name == "posix":
            os.chmod(filename, os.stat(filename).st_mode | 0o111)

    def _tools_for_version(self):
        ret = {}
        # Select release-upstream from version (wasm-binaries)
        with open(os.path.join(self.source_folder, "emscripten-releases-tags.json")) as f:
            data = json.load(f)
            ret["wasm"] = f"releases-upstream-{data['releases'][self.version]}-64bit"
        # Select python and node versions
        with open(os.path.join(self.source_folder, "emsdk_manifest.json")) as f:
            data = json.load(f)
            tools = data["tools"]
            if self.settings.os == "Windows":
                python = next(
                    (it for it in tools if (it["id"] == "python" and not it.get("is_old", False))),
                    None,
                )
                if python:
                    ret["python"] = f"python-{python['version']}-64bit"
            node = next(
                (it for it in tools if (it["id"] == "node" and not it.get("is_old", False))),
                None,
            )
            if node:
                ret["nodejs"] = f"node-{node['version']}-64bit"
        return ret

    def build(self):
        emsdk_ext = "emsdk.bat" if self._settings_build.os == "Windows" else "emsdk"
        emsdk = os.path.join(self.source_folder, emsdk_ext)
        self._chmod_plus_x(os.path.join(self.source_folder, "emsdk"))

        # Install all required tools (including bundled node, needed by embuilder)
        required_tools = self._tools_for_version()
        for value in required_tools.values():
            self.run(f'"{emsdk}" install {value}', cwd=self.source_folder)
            self.run(f'"{emsdk}" activate {value}', cwd=self.source_folder)

    def package(self):
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))
        copy(self, "*", src=self.source_folder, dst=os.path.join(self.package_folder, "bin"))
        emscripten = os.path.join(self.package_folder, "bin", "upstream", "emscripten")
        toolchain = os.path.join(emscripten, "cmake", "Modules", "Platform", "Emscripten.cmake")
        # Allow Conan to find its own package libraries during cross-compilation
        replace_in_file(
            self, toolchain,
            "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)",
            "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)",
        )
        replace_in_file(
            self, toolchain,
            "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)",
            "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)",
        )
        replace_in_file(
            self, toolchain,
            "set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)",
            "set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)",
        )
        # Ensure ports directory exists so emcc can download ports at build time
        os.makedirs(os.path.join(self._em_cache, "ports"), exist_ok=True)
        if not cross_building(self):
            self.run("embuilder build MINIMAL", env=["conanemsdk", "conanrun"])
            # Force emscripten to accept the cache as-is after relocation
            sanity_file = os.path.join(self._em_cache, "sanity.txt")
            if os.path.exists(sanity_file):
                os.remove(sanity_file)

    def _define_tool_var(self, value):
        suffix = ".bat" if self.settings.os == "Windows" else ""
        path = os.path.join(self._emscripten, f"{value}{suffix}")
        self._chmod_plus_x(path)
        return path

    def package_info(self):
        self.cpp_info.bindirs = self._relative_paths
        self.cpp_info.includedirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.resdirs = []

        # Only inject environment when cross-compiling for Emscripten
        if not hasattr(self, "settings_target") or self.settings_target is None:
            return

        if self.settings_target.os != "Emscripten":
            self.output.warning(
                f"You've added {self.name}/{self.version} as a build requirement, "
                f"while os={self.settings_target.os} != Emscripten"
            )
            return

        toolchain = os.path.join(
            self.package_folder, "bin", "upstream", "emscripten",
            "cmake", "Modules", "Platform", "Emscripten.cmake",
        )
        self.conf_info.prepend("tools.cmake.cmaketoolchain:user_toolchain", toolchain)

        self.buildenv_info.define_path("EMSDK", self._emsdk)
        self.buildenv_info.define_path("EMSCRIPTEN", self._emscripten)

        compiler_executables = {
            "c": self._define_tool_var("emcc"),
            "cpp": self._define_tool_var("em++"),
        }
        self.conf_info.update("tools.build:compiler_executables", compiler_executables)
        self.buildenv_info.define_path("CC", compiler_executables["c"])
        self.buildenv_info.define_path("CXX", compiler_executables["cpp"])
        self.buildenv_info.define_path("AR", self._define_tool_var("emar"))
        self.buildenv_info.define_path("NM", self._define_tool_var("emnm"))
        self.buildenv_info.define_path("RANLIB", self._define_tool_var("emranlib"))
        self.buildenv_info.define_path("STRIP", self._define_tool_var("emstrip"))

        self.cpp_info.builddirs = [
            os.path.join("bin", "releases", "src"),
            os.path.join("bin", "upstream", "emscripten", "cmake", "Modules"),
            os.path.join("bin", "upstream", "emscripten", "cmake", "Modules", "Platform"),
        ]
