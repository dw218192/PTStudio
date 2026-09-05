"""collect build outputs into a package directory."""

from __future__ import annotations

import glob as globmod
import re
import shutil
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

_GLOB_META = set("*?[{")
_BRACE_RE = re.compile(r"\{([^{}]+)\}")


def _expand_braces(pattern: str) -> list[str]:
    """Expand ``{a,b,c}`` brace groups into multiple glob patterns.

    Only single-level braces are common in file-extension patterns
    (e.g. ``*.{js,wasm}``).  Nested braces are handled by recursion.
    """
    match = _BRACE_RE.search(pattern)
    if not match:
        return [pattern]
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    results: list[str] = []
    for alt in match.group(1).split(","):
        results.extend(_expand_braces(prefix + alt.strip() + suffix))
    return results


def _extract_glob_base(pattern: str) -> str:
    """Return the longest directory prefix containing no glob metacharacters.

    >>> _extract_glob_base("/a/b/c/**/*.exe")
    '/a/b/c'
    >>> _extract_glob_base("/a/b/*.dll")
    '/a/b'
    >>> _extract_glob_base("**/*.exe")
    '.'
    """
    parts = pattern.replace("\\", "/").split("/")
    base_parts: list[str] = []
    for part in parts:
        if any(c in part for c in _GLOB_META):
            break
        base_parts.append(part)
    if not base_parts:
        return "."
    base = "/".join(base_parts)
    # If the entire pattern had no globs, return the parent directory.
    if base == pattern.replace("\\", "/"):
        return str(Path(base).parent)
    return base


def run(ctx: ProjectContext, args: dict[str, Any]) -> None:
    args = ctx.arguments(
        "package",
        {"dry_run": False, "no_clean": False, "output_dir": None, "mappings": []},
        args,
    )
    formatter = TokenFormatter(ctx.tokens, ctx.config)

    # Resolve output directory: CLI > config > default
    output_dir_raw = args.get("output_dir")
    if not output_dir_raw:
        output_dir_raw = "{workspace_root}/_package/{platform}"
    output_dir = Path(formatter.resolve(str(output_dir_raw)))
    if not output_dir.is_absolute():
        output_dir = ctx.workspace_root / output_dir

    dry_run = bool(args.get("dry_run"))
    no_clean = bool(args.get("no_clean"))

    mappings = args.get("mappings")
    if not mappings or not isinstance(mappings, list):
        logger.error("No 'mappings' configured in the package section of config.yaml")
        sys.exit(1)

    # Clean output directory
    if not no_clean and output_dir.exists():
        if dry_run:
            logger.info(f"Would clean: {output_dir}")
        else:
            logger.info(f"Cleaning: {output_dir}")
            remove_tree_with_retries(output_dir)

    total_files = 0
    with log_section("Packaging"):
        for i, mapping in enumerate(mappings):
            if not isinstance(mapping, dict) or "src" not in mapping:
                logger.error(f"Mapping [{i}]: missing 'src' key")
                sys.exit(1)

            src_template = mapping["src"]
            dest = formatter.resolve(mapping.get("dest", "."))
            optional = bool(mapping.get("optional", False))

            # Token-expand, then brace-expand, then glob
            src_expanded = formatter.resolve(src_template)
            patterns = _expand_braces(src_expanded)

            matched: list[Path] = []
            for pat in patterns:
                pat = pat.replace("\\", "/")
                matched.extend(
                    Path(p) for p in globmod.glob(pat, recursive=True) if Path(p).is_file()
                )

            # Deduplicate (brace expansion can overlap)
            matched = sorted(set(matched))

            if not matched:
                if optional:
                    logger.info(f"  mapping[{i}]: {src_template!r} -> 0 files (optional, skipped)")
                    continue
                logger.error(
                    f"Mapping [{i}] matched 0 files: {src_template!r}\n"
                    f"  Expanded to: {src_expanded!r}"
                )
                sys.exit(1)

            # Compute glob base for relative path preservation
            glob_base = Path(_extract_glob_base(src_expanded))

            for src_file in matched:
                try:
                    rel = src_file.relative_to(glob_base)
                except ValueError:
                    rel = Path(src_file.name)

                dest_path = output_dir / dest / rel

                if dry_run:
                    logger.info(f"  {src_file} -> {dest_path}")
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_path)

            logger.info(f"  mapping[{i}]: {src_template!r} -> {dest!r} ({len(matched)} files)")
            total_files += len(matched)

    if dry_run:
        logger.info(f"Dry run: {total_files} file(s) would be packaged to {output_dir}")
    else:
        logger.info(f"Packaged {total_files} file(s) to {output_dir}")


@click.command(name="package", help="Collect build outputs into a package directory")
@project_options
@click.option(
    "--dry-run",
    is_flag=True,
    default=None,
    help="Show what would be copied without copying",
)
@click.option(
    "--no-clean",
    is_flag=True,
    default=None,
    help="Do not remove the output directory before packaging",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Override the output directory",
)
@click.pass_context
def main(cli: click.Context, platform: str, build_type: str, **args: Any) -> None:
    context = load_context(platform, args.get("config") or build_type, passthrough=cli.args)
    run(context, args)


if __name__ == "__main__":
    main()
