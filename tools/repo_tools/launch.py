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


def _discover_executables(build_dir: Path) -> list[Path]:
    """Discover executable files in the build directory's bin folder.
    
    Args:
        build_dir: Root build directory containing the bin folder
        
    Returns:
        List of paths to discovered executables
    """
    bin_dir = build_dir / "bin"
    
    # Check if bin directory exists
    if not bin_dir.exists():
        return []
    
    if not bin_dir.is_dir():
        return []
    
    exe_paths = []
    
    try:
        for file in bin_dir.iterdir():
            # Skip if not a file or symlink to a file
            if not file.is_file():
                continue
            
            # On Windows, check for .exe extension (case-insensitive)
            if is_windows():
                if file.suffix.lower() == ".exe":
                    exe_paths.append(file)
            else:
                # On Unix-like systems, check if file is executable
                # Files without extensions are typically executables
                try:
                    if os.access(file, os.X_OK):
                        exe_paths.append(file)
                except OSError:
                    # Skip files we can't check permissions on
                    continue
    except (PermissionError, OSError) as e:
        # If we can't read the directory, return what we have so far
        print_tool(f"Warning: Could not fully scan {bin_dir}: {e}")
    
    return exe_paths


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
    exe_paths = _discover_executables(build_dir)

    if not exe_paths:
        print_tool(f"No executables found in build directory: {build_dir}")
        print_tool("Build the project first: .\\pts.cmd build")
        sys.exit(1)

    target_exe_path = None
    for exe_path in exe_paths:
        if exe_path.stem == args.executable:
            target_exe_path = exe_path
            break

    if target_exe_path is None:
        print_tool(f"Executable not found: {args.executable}")
        print_tool("Available executables:")
        for exe in exe_paths:
            print_tool(f"  {exe.stem}")
        sys.exit(1)

    env_vars = resolve_env_vars(config, context)
    env = apply_env_overrides(os.environ.copy(), env_vars)
    cmd = [str(target_exe_path)]
    cmd.extend(getattr(args, "passthrough_args", []))
    subprocess.run(cmd, check=True, env=env)


def register_launch_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the launch subcommand."""
    parser = subparsers.add_parser(
        "launch", help="Set up environment and launch executables"
    )
    parser.add_argument(
        "executable",
        type=str,
        default="editor",
        help="Executable to launch (default: editor)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str.casefold,
        choices=["debug", "release", "relwithdebinfo", "minsizerel"],
        default="debug",
        help="Build configuration to launch (default: debug)",
    )
    parser.set_defaults(func=launch_command)
