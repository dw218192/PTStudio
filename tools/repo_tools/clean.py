"""Clean subcommand implementation."""

import argparse
from pathlib import Path

from repo_tools import (
    RepoContext,
    RepoTool,
    build_repo_context,
    load_repo_config,
    logger,
)
from repo_tools.build import _remove_tree_with_retries


def clean_command(args: argparse.Namespace) -> None:
    root = Path(__file__).parent.parent.parent
    config = load_repo_config(root)
    context = build_repo_context(root, args.build_type, config, args.platform)

    build_root = Path(context["build_root"])
    build_dir = Path(context["build_dir"])
    conan_deps_root = Path(context["conan_deps_root"])

    targets: list[Path] = []

    if args.all:
        targets.append(build_root)
    else:
        targets.append(build_dir)
        if args.deps:
            targets.append(conan_deps_root)

    if args.locks:
        for lock in root.glob("conan*.lock"):
            targets.append(lock)

    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique_targets: list[Path] = []
    for t in targets:
        resolved = t.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_targets.append(t)

    if not unique_targets:
        logger.info("Nothing to clean.")
        return

    for target in unique_targets:
        if not target.exists():
            logger.info(f"Skip (not found): {target}")
            continue
        if args.dry_run:
            logger.info(f"Would remove: {target}")
        else:
            logger.info(f"Removing: {target}")
            if target.is_file():
                target.unlink()
            else:
                _remove_tree_with_retries(target)

    if args.dry_run:
        logger.info("Dry run complete. No files were removed.")
    else:
        logger.info("Clean complete.")


class CleanTool(RepoTool):
    name = "clean"
    help = "Remove build artifacts and caches"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--deps",
            action="store_true",
            help="Also remove deployed dependencies for the current platform",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Remove entire _build/ directory (all platforms and configurations)",
        )
        parser.add_argument(
            "--locks",
            action="store_true",
            help="Remove conan*.lock files from the repository root",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be removed without deleting anything",
        )
        parser.add_argument(
            "--build-type",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration type (default: Debug)",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(
            platform=context["platform"],
            build_type=context["build_type"],
            deps=False,
            all=False,
            locks=False,
            dry_run=False,
        )

    def execute(self, args: argparse.Namespace) -> None:
        clean_command(args)
