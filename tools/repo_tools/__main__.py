"""Main entry point for repo tools."""

import argparse
import importlib
import inspect
import pkgutil
from pathlib import Path

from repo_tools import (
    apply_repo_tool_args,
    build_repo_context,
    create_repo_tool_args,
    get_repo_tool_config_args,
    infer_build_type,
    load_repo_config,
    logger,
    register_repo_tool_parser,
    detect_platform_identifier,
    RepoTool,
)


def _discover_tools() -> list[RepoTool]:
    tools: list[RepoTool] = []
    package = importlib.import_module("repo_tools")
    for module_info in pkgutil.iter_modules(package.__path__):
        name = module_info.name
        if name.startswith("_") or name == "__main__":
            continue
        module = importlib.import_module(f"{package.__name__}.{name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is RepoTool or not issubclass(cls, RepoTool):
                continue
            if cls.__module__ != module.__name__:
                continue
            try:
                tool = cls()
            except Exception as exc:
                logger.warning(
                    f"Skipping repo tool '{cls.__name__}' due to init error: {exc}"
                )
                continue
            if not tool.name:
                logger.warning(
                    f"Skipping repo tool '{cls.__name__}' with empty name"
                )
                continue
            tools.append(tool)
    return sorted(tools, key=lambda entry: entry.name)


def _create_common_parser() -> argparse.ArgumentParser:
    """Create a parent parser with arguments shared across all subcommands."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--platform",
        default=None,
        help="Platform identifier (e.g., windows-x64, linux-x64, wasm). Auto-detected by default.",
    )
    return parent


def main() -> None:
    """Main entry point."""
    common_parser = _create_common_parser()
    parser = argparse.ArgumentParser(
        description="PTStudio repository tools",
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Register tooling commands
    for tool in _discover_tools():
        register_repo_tool_parser(subparsers, tool, common_parser)

    args, unknown_args = parser.parse_known_args()
    root = Path(__file__).parent.parent.parent
    config = load_repo_config(root)
    tool_name = args.command
    config_args = get_repo_tool_config_args(config, tool_name)
    build_type = infer_build_type(tool_name, args, config_args)

    # Detect platform identifier (top-level, available to all tools)
    platform_id = detect_platform_identifier(
        platform_override=args.platform
    )

    context = build_repo_context(root, build_type, config, platform_id)

    merged_args = create_repo_tool_args(tool_name, context)
    apply_repo_tool_args(merged_args, config_args)
    cli_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "func", "platform"}  # platform already handled via context
    }
    apply_repo_tool_args(merged_args, cli_args)
    if unknown_args:
        merged_args.passthrough_args = unknown_args
    elif not hasattr(merged_args, "passthrough_args"):
        merged_args.passthrough_args = []
    args.func(merged_args)


if __name__ == "__main__":
    main()
