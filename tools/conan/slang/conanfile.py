from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy, get
import os


class SlangConan(ConanFile):
    name = "slang"
    version = "2026.1"
    package_type = "application"
    settings = "os", "arch"
    license = "Apache-2.0"
    description = "Slang shading language compiler and tools"
    homepage = "https://shader-slang.org/"
    no_copy_source = True

    def _arch_tag(self) -> str:
        arch = str(self.settings.arch)
        if arch in ("x86_64", "amd64"):
            return "x86_64"
        if arch in (
            "armv8",
            "armv8.2",
            "armv8.3",
            "armv8.4",
            "armv8.5",
            "armv8.6",
            "armv8.7",
            "armv8.8",
            "aarch64",
        ):
            return "aarch64"
        raise ConanInvalidConfiguration(f"Unsupported architecture for Slang: {arch}")

    def _platform_tag(self) -> str:
        os_name = str(self.settings.os)
        arch_tag = self._arch_tag()
        if os_name == "Windows":
            return f"windows-{arch_tag}"
        if os_name == "Linux":
            return f"linux-{arch_tag}"
        if os_name == "Macos":
            return f"macos-{arch_tag}"
        raise ConanInvalidConfiguration(f"Unsupported OS for Slang: {os_name}")

    def build(self) -> None:
        platform_tag = self._platform_tag()
        archive_name = f"slang-{self.version}-{platform_tag}.zip"
        url = (
            "https://github.com/shader-slang/slang/releases/download/"
            f"v{self.version}/{archive_name}"
        )
        get(self, url=url, destination=self.build_folder)

    def _slang_root(self) -> str:
        candidates = [self.build_folder]
        for entry in os.listdir(self.build_folder):
            candidate = os.path.join(self.build_folder, entry)
            if os.path.isdir(candidate):
                candidates.append(candidate)
        for candidate in candidates:
            if os.path.isdir(os.path.join(candidate, "bin")):
                return candidate
        raise RuntimeError(f"No 'bin' directory found in {self.build_folder}")

    def package(self) -> None:
        copy(self, "*", src=self._slang_root(), dst=self.package_folder)

    def package_info(self) -> None:
        self.cpp_info.bindirs = ["bin"]
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.includedirs = ["include"]
