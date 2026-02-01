"""Shared utilities for repo tools."""

import argparse
import functools
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
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
    deps_root: str
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


def get_repo_tool_name(tool: RepoTool) -> str | None:
    return tool.name if tool.name else None


def create_repo_tool_args(name: str, context: RepoContext) -> argparse.Namespace:
    tool = get_repo_tool(name)
    if tool is None:
        raise KeyError(f"Repo tool '{name}' is not registered.")
    return tool.default_args(context)


def list_repo_tools() -> list[str]:
    return sorted(_REPO_TOOL_REGISTRY.keys())


def register_repo_tool_parser(
    subparsers: argparse._SubParsersAction, tool: RepoTool
) -> None:
    parser = subparsers.add_parser(
        tool.name, help=tool.help, argument_default=argparse.SUPPRESS
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


def _get_optional_config_value(config: dict, key_path: str) -> str | None:
    current = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return str(current) if current is not None else None


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

    # Add deps_root for referencing deployed dependencies
    deps_root = Path(resolved_conan_deps_root) / "full_deploy" / "host"

    context: RepoContext = {
        **template_context,
        "conan_lock": resolved_conan_lock,
        "deps_root": str(deps_root),
        "build_dir": str(build_root / build_type),
    }
    return context


def resolve_env_vars(env_config: dict | None, context: RepoContext) -> dict:
    """Resolve environment variables, expanding glob patterns."""
    import glob

    if not env_config:
        return {}
    if "env" in env_config and isinstance(env_config.get("env"), dict):
        env_config = env_config.get("env", {})
    resolved = {}
    for key, value in env_config.items():
        if isinstance(value, list):
            expanded_values = []
            for item in value:
                resolved_item = _resolve_template(str(item), context)
                # Expand glob pattern - take the first match if multiple found
                matches = sorted(glob.glob(resolved_item))
                expanded_values.append(matches[0] if matches else resolved_item)
            resolved[key] = os.pathsep.join(expanded_values)
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
