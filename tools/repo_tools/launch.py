"""Launch subcommand implementation."""

import argparse
import subprocess
import sys
from pathlib import Path


def _editor_executable_path(root: Path, config: str) -> Path:
    exe_name = "editor.exe" if sys.platform == "win32" else "editor"
    return root / "_build" / config / "bin" / exe_name


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
    config = _normalize_config(args.config)
    exe_path = _editor_executable_path(root, config)

    if not exe_path.exists():
        print(f"Editor executable not found: {exe_path}")
        print("Build the project first: .\\pts.cmd build")
        sys.exit(1)

    cmd = [str(exe_path)]
    cmd.extend(getattr(args, "passthrough_args", []))
    subprocess.run(cmd, check=True)


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
