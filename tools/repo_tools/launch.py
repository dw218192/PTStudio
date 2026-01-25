"""Launch subcommand implementation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from repo_tools import (
    apply_env_overrides,
    build_repo_context,
    is_windows,
    load_repo_config,
    print_tool,
    resolve_env_vars,
)


def _editor_executable_path(build_dir: Path) -> Path:
    exe_name = "editor.exe" if is_windows() else "editor"
    return build_dir / "bin" / exe_name


def _normalize_config(config: str) -> str:
    config_map = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "minsizerel": "MinSizeRel",
    }
    return config_map[config.casefold()]


def launch_command(args: argparse.Namespace) -> None:
    """Launch the editor executable for native deployment."""
    root = Path(__file__).parent.parent.parent
    build_type = _normalize_config(args.config)
    config = load_repo_config(root)
    context = build_repo_context(root, build_type, config)
    build_dir = Path(context["build_dir"])
    exe_path = _editor_executable_path(build_dir)

    if not exe_path.exists():
        print_tool(f"Editor executable not found: {exe_path}")
        print_tool("Build the project first: .\\pts.cmd build")
        sys.exit(1)

    env_vars = resolve_env_vars(config, context)
    env = apply_env_overrides(os.environ.copy(), env_vars)

    cmd = [str(exe_path)]
    cmd.extend(getattr(args, "passthrough_args", []))
    subprocess.run(cmd, check=True, env=env)


def register_launch_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the launch subcommand."""
    parser = subparsers.add_parser("launch", help="Launch the editor executable")
    parser.add_argument(
        "-c",
        "--config",
        type=str.casefold,
        choices=["debug", "release", "relwithdebinfo", "minsizerel"],
        default="Release",
        help="Build configuration to launch (default: Release)",
    )
    parser.set_defaults(func=launch_command)
