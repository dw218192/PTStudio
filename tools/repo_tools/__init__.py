"""Shared utilities for repo tools."""

import argparse
import contextlib
import functools
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import Optional, TypedDict

from colorama import Fore, Style, init as colorama_init
import yaml

colorama_init()


class RepoContext(TypedDict):
    workspace_root: str
    build_root: str
    logs_root: str
    platform: str
    build_type: str
    conan_deps_root: str
    conan_lock: str
    build_dir: str


class RepoTool:
    name: str = ""
    help: str = ""

    def setup(self, parser: argparse.ArgumentParser) -> None:
        """Define supported CLI args for the tool."""
        return None

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        """Return default args for the tool (merged before config/CLI)."""
        return argparse.Namespace()

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the tool with merged args."""
        raise NotImplementedError


_REPO_TOOL_REGISTRY: dict[str, RepoTool] = {}


def register_repo_tool(tool: RepoTool) -> None:
    if not tool.name:
        raise ValueError("Repo tool name cannot be empty.")
    existing = _REPO_TOOL_REGISTRY.get(tool.name)
    if existing:
        if existing is tool:
            return
        raise ValueError(f"Repo tool '{tool.name}' is already registered.")
    _REPO_TOOL_REGISTRY[tool.name] = tool


def get_repo_tool(name: str) -> RepoTool | None:
    return _REPO_TOOL_REGISTRY.get(name)


def create_repo_tool_args(name: str, context: RepoContext) -> argparse.Namespace:
    tool = get_repo_tool(name)
    if tool is None:
        raise KeyError(f"Repo tool '{name}' is not registered.")
    args = tool.default_args(context)
    if not hasattr(args, "passthrough_args"):
        args.passthrough_args = []
    return args


def invoke_tool(
    name: str,
    context: RepoContext,
    config: dict,
    extra_args: dict | None = None,
) -> None:
    """Invoke a registered repo tool programmatically."""
    tool = get_repo_tool(name)
    if tool is None:
        raise KeyError(f"Repo tool '{name}' is not registered.")
    args = create_repo_tool_args(name, context)
    config_args = get_repo_tool_config_args(config, name)
    apply_repo_tool_args(args, config_args)
    if extra_args:
        step_args = normalize_repo_tool_args(extra_args)
        apply_repo_tool_args(args, step_args)
    tool.execute(args)


def register_repo_tool_parser(
    subparsers: argparse._SubParsersAction,
    tool: RepoTool,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    parents = [parent_parser] if parent_parser else []
    parser = subparsers.add_parser(
        tool.name, help=tool.help, argument_default=argparse.SUPPRESS, parents=parents
    )
    tool.setup(parser)
    parser.set_defaults(func=tool.execute)
    register_repo_tool(tool)


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


def _is_ci() -> bool:
    """Return True when running inside GitHub Actions."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


@contextlib.contextmanager
def log_section(title: str) -> Generator[None, None, None]:
    """Foldable CI section (``::group::``) or styled terminal header."""
    if _is_ci():
        print(f"::group::{title}", flush=True)
    else:
        logger.info(f"── {title} ──")
    try:
        yield
    finally:
        if _is_ci():
            print("::endgroup::", flush=True)


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


def run_command(
    cmd: list[str],
    log_file: Optional[Path] = None,
    env_script: Optional[Path] = None,
) -> None:
    """Run a command and optionally tee output to a log file.

    If *env_script* is provided and the file exists, the command is executed
    inside a shell that sources the script first (``call`` on Windows,
    ``source`` on POSIX).  Python's own ``os.environ`` is **not** modified.
    """
    use_shell = False
    run_cmd: list[str] | str = cmd
    if env_script is not None:
        script = env_script
        if not script.suffix:
            script = script.with_suffix(".bat" if is_windows() else ".sh")
        if script.exists():
            cmd_str = subprocess.list2cmdline(cmd)
            if is_windows():
                run_cmd = f'call "{script}" >nul 2>&1 && {cmd_str}'
            else:
                run_cmd = f'source "{script}" >/dev/null 2>&1 && {cmd_str}'
            use_shell = True

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            process = subprocess.Popen(
                run_cmd,
                shell=use_shell,
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
        subprocess.run(run_cmd, shell=use_shell, check=True)


def ensure_conan_profile() -> None:
    """Ensure Conan profiles exist, run detect if needed."""
    profile_dir = Path.home() / ".conan2" / "profiles"

    if not profile_dir.exists() or not any(profile_dir.iterdir()):
        logger.info("No Conan profiles found. Running 'conan profile detect'...")
        conan_exe = find_venv_executable("conan")
        subprocess.run([conan_exe, "profile", "detect"], check=True)
    else:
        logger.info("Conan profiles already exist.")


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


def _get_optional_config_value(config: dict, key_path: str) -> str | None:
    current = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return str(current) if current is not None else None


def _resolve_template(value: str, context: RepoContext) -> str:
    try:
        return value.format(**context)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "unknown"
        raise KeyError(f"Missing config template value: {missing}") from exc


def resolve_path(root: Path, template: str, context: RepoContext) -> Path:
    resolved = _resolve_template(template, context)
    path = Path(resolved)
    if not path.is_absolute():
        path = root / path
    return path


def detect_platform_identifier(
    platform_override: str | None = None,
    conan_profile_path: Path | None = None
) -> str:
    """Detect platform identifier for build directory structure.

    Platform detection priority:
    1. Explicit --platform override
    2. Parse from Conan profile
    3. Auto-detect host platform
    """
    # 1. Explicit override
    if platform_override:
        return platform_override

    # 2. Parse from Conan profile
    if conan_profile_path and conan_profile_path.exists():
        try:
            profile_content = conan_profile_path.read_text()
            os_match = re.search(r'^os=(\w+)', profile_content, re.MULTILINE)
            arch_match = re.search(r'^arch=(\w+)', profile_content, re.MULTILINE)
            if os_match and arch_match:
                os_val = os_match.group(1)
                arch_val = arch_match.group(1)
                return _map_platform_identifier(os_val, arch_val)
        except Exception:
            pass  # Fall through to host detection

    # 4. Host platform auto-detection
    system = platform.system()
    machine = platform.machine().lower()

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64", "armv8"):
        arch = "arm64"
    else:
        arch = machine

    # Map system
    if system == "Windows":
        return f"windows-{arch}"
    elif system == "Linux":
        return f"linux-{arch}"
    elif system == "Darwin":
        return f"macos-{arch}"
    else:
        return f"{system.lower()}-{arch}"


def _map_platform_identifier(os_val: str, arch_val: str) -> str:
    """Map Conan os/arch settings to platform identifier."""
    if os_val == "Emscripten" and arch_val == "wasm":
        return "emscripten"

    # Normalize OS
    os_map = {
        "Windows": "windows",
        "Linux": "linux",
        "Macos": "macos",
        "Darwin": "macos",
    }
    os_normalized = os_map.get(os_val, os_val.lower())

    # Normalize arch
    arch_map = {
        "x86_64": "x64",
        "x86": "x86",
        "armv8": "arm64",
        "armv8_32": "arm",
        "wasm": "wasm",
    }
    arch_normalized = arch_map.get(arch_val, arch_val.lower())

    return f"{os_normalized}-{arch_normalized}"


def is_platform_compatible(target_platform: str, host_platform: str | None = None) -> bool:
    """Check if target platform binaries can run on host platform.

    Args:
        target_platform: Platform identifier of the build (e.g., "emscripten", "windows-x64")
        host_platform: Platform identifier of the host (auto-detected if None)

    Returns:
        True if target can run on host, False otherwise
    """
    if host_platform is None:
        host_platform = detect_platform_identifier()

    # Same platform is always compatible
    if target_platform == host_platform:
        return True

    # Emscripten cannot run natively (needs browser/Node.js)
    if target_platform == "emscripten":
        return False

    # Cross-platform builds cannot run natively
    return False


def build_repo_context(root: Path, build_type: str, config: dict, platform_id: str) -> RepoContext:
    build_root_value = _get_optional_config_value(config, "repo_paths.build_root")
    if build_root_value is None:
        build_root_value = _get_config_value(config, "paths.build_root", "_build")
    build_root = root / build_root_value

    logs_root_value = _get_optional_config_value(config, "repo_paths.logs_root")
    if logs_root_value is None:
        logs_root_value = _get_config_value(config, "paths.logs_root", "_logs")
    logs_root = root / logs_root_value

    base_context = {
        "workspace_root": str(root),
        "build_root": str(build_root),
        "logs_root": str(logs_root),
        "platform": platform_id,
        "build_type": build_type,
    }

    conan_deps_root = _get_optional_config_value(
        config, "repo_paths.conan_deps_root"
    )
    if conan_deps_root is None:
        conan_deps_root = _get_config_value(
            config, "paths.conan_deps_root", "{build_root}/deps"
        )
    resolved_conan_deps_root = _resolve_template(conan_deps_root, base_context)
    template_context = {**base_context, "conan_deps_root": resolved_conan_deps_root}

    conan_lock = _get_optional_config_value(config, "repo_paths.conan_lock")
    if conan_lock is None:
        conan_lock = _get_config_value(config, "paths.conan_lock", "conan_glfw.lock")
    resolved_conan_lock = str(resolve_path(root, conan_lock, template_context))

    context: RepoContext = {
        **template_context,
        "conan_lock": resolved_conan_lock,
        "build_dir": str(build_root / platform_id / build_type),
    }
    return context


def load_conan_env(build_dir: Path, preset_type: str = "test") -> dict[str, str]:
    """Load environment variables from Conan-generated CMakePresets.json.

    Args:
        build_dir: The build configuration directory (e.g., _build/windows-x64/Debug/).
        preset_type: Which preset to read:
            "test" for runtime env (DLL paths for launching/testing),
            "configure" for build env (slang, emsdk, etc.).

    Returns:
        Dict of env var name -> resolved value. ``$penv{VAR}`` references are
        replaced with the current process environment value.
    """
    import json

    presets_path = build_dir / "CMakePresets.json"
    if not presets_path.exists():
        logger.warning(f"CMakePresets.json not found: {presets_path}")
        return {}

    presets = json.loads(presets_path.read_text(encoding="utf-8"))

    key = {
        "test": "testPresets",
        "configure": "configurePresets",
        "build": "buildPresets",
    }.get(preset_type, f"{preset_type}Presets")

    preset_list = presets.get(key, [])
    if not preset_list:
        logger.warning(f"No {key} found in {presets_path}")
        return {}

    env = preset_list[0].get("environment", {})
    if not env:
        return {}

    resolved: dict[str, str] = {}
    for var, value in env.items():
        if not isinstance(value, str):
            continue
        result = re.sub(
            r"\$penv\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
        resolved[var] = result
    return resolved


def normalize_env_config(env_value: object) -> dict[str, object]:
    if not env_value:
        return {}
    if isinstance(env_value, dict):
        return dict(env_value)
    if isinstance(env_value, str):
        env_items = [env_value]
    elif isinstance(env_value, list):
        env_items = env_value
    else:
        logger.warning(f"Skipping env config with unsupported type: {type(env_value)}")
        return {}

    parsed: dict[str, object] = {}
    for item in env_items:
        if not isinstance(item, str):
            logger.warning(f"Skipping env entry (not a string): {item!r}")
            continue
        text = item.strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            parsed[key] = value
        else:
            parsed[text] = ""
    return parsed


def normalize_build_type(value: str | None) -> str:
    if not value:
        return "Debug"
    normalized = str(value)
    mapping = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "minsizerel": "MinSizeRel",
    }
    return mapping.get(normalized.casefold(), normalized)


def normalize_repo_tool_args(args_value: object) -> dict[str, object]:
    if not args_value:
        return {}
    if isinstance(args_value, dict):
        normalized: dict[str, object] = {}
        for key, value in args_value.items():
            normalized_key = str(key).replace("-", "_")
            normalized[normalized_key] = value
        return normalized
    if isinstance(args_value, str):
        args_list = [args_value]
    elif isinstance(args_value, list):
        args_list = args_value
    else:
        logger.warning(
            f"Skipping repo tool args with unsupported type: {type(args_value)}"
        )
        return {}

    parsed: dict[str, object] = {}
    for arg in args_list:
        if not isinstance(arg, str):
            logger.warning(
                f"Skipping repo tool arg entry (not a string): {arg!r}"
            )
            continue
        text = arg.strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            normalized_key = key.lstrip("-").replace("-", "_")
            parsed[normalized_key] = value
        elif text.startswith("-"):
            normalized_key = text.lstrip("-").replace("-", "_")
            parsed[normalized_key] = True
        else:
            logger.warning(
                f"Skipping repo tool arg entry (expected --flag or key=value): {text}"
            )
    return parsed


def apply_repo_tool_args(target: argparse.Namespace, args: Mapping[str, object]) -> None:
    for key, value in args.items():
        setattr(target, key, value)


def get_repo_tool_config_args(config: dict, tool_name: str) -> dict[str, object]:
    tool_config = config.get(tool_name)
    if tool_config is None:
        for section_key in ("repo_tools", "tools"):
            section = config.get(section_key, {})
            if not isinstance(section, dict):
                continue
            tool_config = section.get(tool_name)
            if tool_config is not None:
                break

    if tool_config is None:
        return {}
    if isinstance(tool_config, (dict, list, str)):
        return normalize_repo_tool_args(tool_config)
    logger.warning(
        f"Skipping repo tool config with unsupported type: {type(tool_config)}"
    )
    return {}


def infer_build_type(
    tool_name: str, cli_args: argparse.Namespace, config_args: Mapping[str, object]
) -> str:
    if tool_name == "launch":
        value = getattr(cli_args, "config", None) or config_args.get("config")
        return normalize_build_type(str(value)) if value else "Debug"
    value = getattr(cli_args, "build_type", None) or config_args.get("build_type")
    return normalize_build_type(str(value)) if value else "Debug"
