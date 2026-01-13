"""Package subcommand implementation."""

import argparse
import shutil
import sys
from pathlib import Path

from repo_tools import logger


def package_command(args: argparse.Namespace) -> None:
    """Package command - deploys plugins and resources to output directory."""
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "_build" / args.build_type
    bin_dir = build_dir / "bin"
    plugins_output_dir = bin_dir / "plugins"

    if not build_dir.exists():
        logger.error(f"Build directory does not exist: {build_dir}")
        logger.error(f"Run 'pts build --build-type={args.build_type}' first")
        sys.exit(1)

    if not bin_dir.exists():
        logger.error(f"Bin directory does not exist: {bin_dir}")
        sys.exit(1)

    logger.info(f"Packaging plugins for {args.build_type} configuration")
    logger.info(f"  Source: {build_dir}")
    logger.info(f"  Target: {plugins_output_dir}")

    # Create plugins output directory
    plugins_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all plugin DLLs in the build directory
    plugin_dirs = (
        list((repo_root / "plugins").iterdir())
        if (repo_root / "plugins").exists()
        else []
    )

    plugins_copied = 0
    for plugin_dir in plugin_dirs:
        if not plugin_dir.is_dir():
            continue

        plugin_name = plugin_dir.name

        # Look for built plugin DLL in build directory
        # Plugins are typically built to build/<config>/bin/<plugin_name>.(dll|so|dylib)
        plugin_dll = None

        for ext in [".dll", ".so", ".dylib"]:
            candidate = bin_dir / f"{plugin_name}{ext}"
            if candidate.exists():
                plugin_dll = candidate
                break

        if not plugin_dll:
            logger.debug(f"  No built DLL found for plugin: {plugin_name}")
            continue

        # Copy plugin DLL to plugins output directory
        dest = plugins_output_dir / plugin_dll.name
        try:
            shutil.copy2(plugin_dll, dest)
            logger.info(f"  Copied: {plugin_dll.name} -> plugins/")
            plugins_copied += 1
        except Exception as e:
            logger.error(f"  Failed to copy {plugin_dll.name}: {e}")

    if plugins_copied == 0:
        logger.warning("No plugins were packaged")
        logger.info(
            "Hint: Build plugin projects first, they will appear in the bin/ directory"
        )
    else:
        logger.info(f"Successfully packaged {plugins_copied} plugin(s)")


def register_package_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the package subcommand."""
    parser = subparsers.add_parser(
        "package", help="Package plugins and resources for distribution"
    )
    parser.add_argument(
        "--build-type",
        default="Debug",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        help="Build configuration type (default: Debug)",
    )
    parser.set_defaults(func=package_command)
