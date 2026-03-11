"""Clean subcommand — removes build artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    ToolContext,
    logger,
    remove_tree_with_retries,
)


class CleanTool(RepoTool):
    name = "clean"
    help = "Remove build artifacts"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "--dry-run",
            is_flag=True,
            default=None,
            help="Print what would be deleted without deleting",
        )(cmd)
        cmd = click.option(
            "--deps",
            is_flag=True,
            default=None,
            help="Also delete deployed Conan dependencies",
        )(cmd)
        cmd = click.option(
            "--all",
            is_flag=True,
            default=None,
            help="Delete entire _build/ and _logs/ directories",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "dry_run": False,
            "deps": False,
            "all": False,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        dry_run: bool = bool(args.get("dry_run"))
        clean_deps: bool = bool(args.get("deps"))
        clean_all: bool = bool(args.get("all"))

        targets: list[Path] = []

        if clean_all:
            targets.append(Path(ctx.tokens["build_root"]))
            targets.append(Path(ctx.tokens["logs_root"]))
        else:
            build_type = ctx.dimensions.get("build_type")
            platform = ctx.dimensions.get("platform", "")

            if build_type:
                targets.append(Path(ctx.tokens["build_dir"]))
            else:
                platform_dir = Path(ctx.tokens["build_root"]) / platform
                targets.append(platform_dir)

            targets.append(Path(ctx.tokens["logs_root"]))

            if clean_deps:
                targets.append(Path(ctx.tokens["conan_deps_root"]))

        for target in targets:
            if target.exists():
                if dry_run:
                    logger.info(f"Would delete: {target}")
                else:
                    logger.info(f"Deleting: {target}")
                    remove_tree_with_retries(target)
            else:
                logger.info(f"Skipping (does not exist): {target}")

        if dry_run:
            logger.info("Dry run — nothing was deleted.")
        else:
            logger.info("Clean complete.")
