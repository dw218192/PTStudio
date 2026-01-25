"""Shared utilities for repo tools."""

import functools
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Optional, TypedDict

from colorama import Fore, Style, init as colorama_init
import yaml

colorama_init()


class RepoContext(TypedDict):
    workspace_root: str
    build_root: str
    logs_root: str
    build_type: str
    conan_deps_root: str
    conan_lock: str
    usd_install_dir: str
    usd_build_dir: str
    build_dir: str


def _level_color(levelno: int) -> str:
    if levelno >= logging.ERROR:
        return Fore.RED
    if levelno >= logging.WARNING:
        return Fore.YELLOW
    return Fore.CYAN


class ToolFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _level_color(record.levelno)
        message = record.getMessage()
        return f"{color}[{record.levelname.lower()}]{Style.RESET_ALL} {message}"


logger = logging.getLogger("repo_tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ToolFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def print_tool(message: str) -> None:
    print(f"{Fore.CYAN}[pts]{Style.RESET_ALL} {message}", flush=True)


def print_subprocess_line(line: str) -> None:
    text = line.rstrip()
    print(f"{Style.DIM}{text}{Style.RESET_ALL}")


@functools.cache
def find_venv_executable(name: str) -> str:
    """Find an executable in the virtual environment, fallback to system PATH."""
    # Get the Scripts/bin directory relative to the Python executable
    python_exe = Path(sys.executable)
    scripts_dir = python_exe.parent
    exe_path = scripts_dir / (name + (".exe" if sys.platform == "win32" else ""))

    if exe_path.exists():
        return str(exe_path)

    # Fallback to system PATH
    exe_path = shutil.which(name)
    if exe_path:
        return exe_path

    logger.warning(f"Executable {name} not found in virtual environment or system PATH")
    return name


def run_command(cmd: list[str], log_file: Optional[Path] = None) -> None:
    """Run a command and optionally tee output to a log file."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in process.stdout:
                print_subprocess_line(line)
                f.write(line)
            process.wait()
            if process.returncode != 0:
                sys.exit(process.returncode)
    else:
        subprocess.run(cmd, check=True)


def augment_env_with_usd(env: dict, usd_install_dir: Path) -> dict:
    if usd_install_dir.exists():
        usd_bin = usd_install_dir / "bin"
        usd_lib = usd_install_dir / "lib"
        env["PATH"] = f"{usd_bin}{os.pathsep}{usd_lib}{os.pathsep}{env.get('PATH', '')}"
    else:
        logger.error(f"USD install directory does not exist: {usd_install_dir}")
    return env


def ensure_conan_profile() -> None:
    """Ensure Conan profiles exist, run detect if needed."""
    profile_dir = Path.home() / ".conan2" / "profiles"

    if not profile_dir.exists() or not any(profile_dir.iterdir()):
        print_tool("No Conan profiles found. Running 'conan profile detect'...")
        conan_exe = find_venv_executable("conan")
        subprocess.run([conan_exe, "profile", "detect"], check=True)
    else:
        print_tool("Conan profiles already exist.")


def load_repo_config(root: Path) -> dict:
    config_path = root / "config.yaml"
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("config.yaml must contain a top-level mapping.")
    return data


def _get_config_value(config: dict, key_path: str, default: str) -> str:
    current = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return str(current) if current is not None else default


def _resolve_template(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format(**context)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "unknown"
        raise KeyError(f"Missing config template value: {missing}") from exc


def resolve_path(root: Path, template: str, context: Mapping[str, str]) -> Path:
    resolved = _resolve_template(template, context)
    path = Path(resolved)
    if not path.is_absolute():
        path = root / path
    return path


def build_repo_context(root: Path, build_type: str, config: dict) -> RepoContext:
    build_root = root / _get_config_value(config, "paths.build_root", "_build")
    logs_root = root / _get_config_value(config, "paths.logs_root", "_logs")

    base_context = {
        "workspace_root": str(root),
        "build_root": str(build_root),
        "logs_root": str(logs_root),
        "build_type": build_type,
    }

    conan_deps_root = _get_config_value(
        config, "paths.conan_deps_root", "{build_root}/deps"
    )
    resolved_conan_deps_root = _resolve_template(conan_deps_root, base_context)
    template_context = {**base_context, "conan_deps_root": resolved_conan_deps_root}

    conan_lock = _get_config_value(config, "paths.conan_lock", "conan_glfw.lock")
    resolved_conan_lock = str(resolve_path(root, conan_lock, template_context))

    usd_install = _get_config_value(
        config, "paths.usd_install_dir", "{build_root}/usd/{build_type}/install"
    )
    resolved_usd_install_dir = str(resolve_path(root, usd_install, template_context))

    usd_build = _get_config_value(
        config, "paths.usd_build_dir", "{build_root}/usd/{build_type}/build"
    )
    resolved_usd_build_dir = str(resolve_path(root, usd_build, template_context))

    context: RepoContext = {
        **template_context,
        "conan_lock": resolved_conan_lock,
        "usd_install_dir": resolved_usd_install_dir,
        "usd_build_dir": resolved_usd_build_dir,
        "build_dir": str(build_root / build_type),
    }
    return context


def resolve_env_vars(config: dict, context: RepoContext) -> dict:
    env_config = config.get("env", {})
    resolved = {}
    for key, value in env_config.items():
        if isinstance(value, list):
            values = [_resolve_template(str(item), context) for item in value]
            resolved[key] = os.pathsep.join(values)
        else:
            resolved[key] = _resolve_template(str(value), context)
    return resolved


def apply_env_overrides(env: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key.upper() == "PATH":
            if is_windows():  # case-insensitive on Windows
                key = next((k for k in env if k.upper() == "PATH"), key)
            env[key] = f"{value}{os.pathsep}{env.get(key, '')}"
        else:
            env[key] = value
    return env


def resolve_slang_shaders(
    root: Path, config: dict, context: RepoContext
) -> list[tuple[Path, Path]]:
    shaders = config.get("slang", {}).get("shaders", [])
    resolved = []
    for shader in shaders:
        if not isinstance(shader, dict):
            continue
        input_value = shader.get("input")
        if not input_value:
            continue
        output_value = shader.get("output")
        input_path = resolve_path(root, str(input_value), context)

        if output_value:
            output_path = resolve_path(root, str(output_value), context)
        else:
            output_path = input_path.with_suffix(".wgsl")

        resolved.append((input_path, output_path))
    return resolved


def resolve_slang_test_output(
    root: Path, config: dict, context: RepoContext
) -> Path | None:
    test_shader = config.get("slang", {}).get("test_shader")
    if not test_shader:
        return None
    return resolve_path(root, str(test_shader), context)


def find_slangc(root: Path, config: dict, context: RepoContext) -> Path | None:
    compiler_path = config.get("slang", {}).get("compiler_path")
    if compiler_path:
        path = resolve_path(root, str(compiler_path), context)
        if path.exists():
            return path

    search_roots = config.get("slang", {}).get("compiler_search_roots", [])
    candidates: list[Path] = []
    for entry in search_roots:
        candidate = resolve_path(root, str(entry), context)
        if candidate.exists():
            candidates.append(candidate)

    exe_name = "slangc.exe" if is_windows() else "slangc"
    for candidate_root in candidates:
        for path in candidate_root.rglob(exe_name):
            return path

    exe_path = shutil.which(exe_name)
    if exe_path:
        return Path(exe_path)
    return None


def compile_slang_shaders(
    root: Path, config: dict, context: RepoContext, logs_dir: Path
) -> None:
    shaders = resolve_slang_shaders(root, config, context)
    if not shaders:
        return

    slangc_path = find_slangc(root, config, context)
    if slangc_path is None:
        raise RuntimeError("slangc not found. Install via Conan or set compiler_path.")

    for input_path, output_path in shaders:
        if not input_path.exists():
            raise FileNotFoundError(f"Slang shader not found: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            if output_path.stat().st_mtime >= input_path.stat().st_mtime:
                continue
        log_file = logs_dir / f"slangc_{input_path.stem}.log"
        run_command(
            [
                str(slangc_path),
                str(input_path),
                "-target",
                "wgsl",
                "-o",
                str(output_path),
            ],
            log_file=log_file,
        )
