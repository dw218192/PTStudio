"""remove build artifacts and temporary files."""

from __future__ import annotations

import glob as globmod
import os
import re
import sys
from pathlib import Path
from typing import Any

import click

from tasks.utils import (
    ProjectContext,
    TokenFormatter,
    log_section,
    logger,
    remove_tree_with_retries,
)
from tasks.utils.project import load_context, project_options

# Always preserve version-control metadata and the active environment.
PROTECTED = {".git", ".pixi", ".agents", ".codex"}

_RE_PREFIX = "re:"


def run(ctx: ProjectContext, args: dict[str, Any]) -> None:
    args = ctx.arguments("clean", {"dry_run": False, "paths": [], "groups": {}}, args)
    formatter = TokenFormatter(ctx.tokens, ctx.config)
    dry_run = bool(args.get("dry_run"))
    groups: dict[str, list[str]] = dict(args.get("groups", {}))
    flat_paths: list[str] = list(args.get("paths", []))
    requested: tuple[str, ...] = args.get("group_names", ())

    if requested:
        patterns: list[str] = []
        for name in requested:
            if name not in groups:
                available = ", ".join(sorted(groups)) if groups else "(none)"
                logger.error(f"Unknown clean group: '{name}'. Available: {available}")
                sys.exit(1)
            patterns.extend(groups[name])
    else:
        patterns = flat_paths[:]
        for group_paths in groups.values():
            patterns.extend(group_paths)

    # Split into regex vs glob patterns
    regex_patterns: list[re.Pattern[str]] = []
    glob_patterns: list[str] = []
    for pat in patterns:
        if pat.startswith(_RE_PREFIX):
            regex_patterns.append(re.compile(pat[len(_RE_PREFIX) :]))
        else:
            glob_patterns.append(pat)

    with log_section("Cleaning"):
        removed = 0
        removed += _clean_globs(ctx, formatter, glob_patterns, dry_run)
        removed += _clean_regex(ctx, regex_patterns, dry_run)

    action = "Would remove" if dry_run else "Removed"
    logger.info(f"{action} {removed} item(s)")


def _clean_globs(
    ctx: ProjectContext,
    formatter: TokenFormatter,
    patterns: list[str],
    dry_run: bool,
) -> int:
    removed = 0
    for path_template in patterns:
        expanded = formatter.resolve(path_template)
        resolved = Path(expanded)
        if not resolved.is_absolute():
            resolved = ctx.workspace_root / resolved

        try:
            resolved.resolve().relative_to(ctx.workspace_root.resolve())
        except ValueError:
            logger.warning(f"Skipping (outside workspace): {resolved}")
            continue

        matched = sorted(globmod.glob(str(resolved), recursive=True))
        if not matched:
            logger.info(f"Not found (skipping): {resolved}")
            continue

        for match in matched:
            removed += _try_remove(ctx, Path(match), dry_run)
    return removed


def _clean_regex(
    ctx: ProjectContext,
    patterns: list[re.Pattern[str]],
    dry_run: bool,
) -> int:
    if not patterns:
        return 0
    removed = 0
    root = ctx.workspace_root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # Use forward slashes for consistent matching
        rel = Path(dirpath).resolve().relative_to(root).as_posix()
        for pat in patterns:
            if pat.search(rel):
                removed += _try_remove(ctx, Path(dirpath), dry_run)
                dirnames.clear()  # don't descend into removed dirs
                break
        # Also check files
        else:
            for fname in filenames:
                file_rel = f"{rel}/{fname}" if rel != "." else fname
                for pat in patterns:
                    if pat.search(file_rel):
                        removed += _try_remove(ctx, Path(dirpath) / fname, dry_run)
                        break
    return removed


def _try_remove(ctx: ProjectContext, p: Path, dry_run: bool) -> int:
    try:
        relative = p.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        logger.warning(f"Skipping (outside workspace): {p}")
        return 0
    if not relative.parts or any(part in PROTECTED for part in relative.parts):
        logger.warning(f"Skipping (protected): {p}")
        return 0
    if p.is_dir():
        if dry_run:
            logger.info(f"Would remove directory: {p}")
        else:
            remove_tree_with_retries(p)
            logger.info(f"Removed directory: {p}")
        return 1
    elif p.is_file():
        if dry_run:
            logger.info(f"Would remove file: {p}")
        else:
            p.unlink()
            logger.info(f"Removed file: {p}")
        return 1
    return 0


@click.command(name="clean", help="Remove build artifacts and temporary files")
@project_options
@click.argument("group_names", nargs=-1)
@click.option(
    "--dry-run",
    is_flag=True,
    default=None,
    help="Show what would be removed",
)
@click.pass_context
def main(cli: click.Context, platform: str, build_type: str, **args: Any) -> None:
    context = load_context(platform, args.get("config") or build_type, passthrough=cli.args)
    run(context, args)


if __name__ == "__main__":
    main()
