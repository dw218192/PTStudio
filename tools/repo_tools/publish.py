"""Publish subcommand — prepare a static site from packaged WASM artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    TokenFormatter,
    ToolContext,
    log_section,
    logger,
    remove_tree_with_retries,
)


def _discover_apps(bin_dir: Path) -> list[str]:
    """Find WASM app stems in bin_dir by scanning for .html files.

    Skips the ``tests/`` subdirectory.
    """
    if not bin_dir.exists() or not bin_dir.is_dir():
        return []

    stems: list[str] = []
    for html in bin_dir.rglob("*.html"):
        if "tests" in html.relative_to(bin_dir).parts:
            continue
        stems.append(html.stem)
    return sorted(set(stems))


def _generate_index(apps: list[str]) -> str:
    """Generate a simple HTML landing page linking to all apps."""
    links = "\n".join(f'    <li><a href="{app}/">{app}</a></li>' for app in apps)
    return f"""\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PTStudio Demos</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 2em auto; }}
    a {{ color: #0366d6; }}
  </style>
</head>
<body>
  <h1>PTStudio Demos</h1>
  <ul>
{links}
  </ul>
</body>
</html>
"""


class PublishTool(RepoTool):
    name = "publish"
    help = "Prepare a static site from packaged WASM artifacts"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "--dry-run",
            is_flag=True,
            default=None,
            help="Show what would be published without copying",
        )(cmd)
        cmd = click.option(
            "--input-dir",
            type=click.Path(),
            default=None,
            help="Override the input directory (default: package bin dir)",
        )(cmd)
        cmd = click.option(
            "--output-dir",
            type=click.Path(),
            default=None,
            help="Override the output directory",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "dry_run": False,
            "input_dir": None,
            "output_dir": None,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        formatter = TokenFormatter(ctx.tokens)

        # Resolve input directory
        input_dir_raw = args.get("input_dir")
        if not input_dir_raw:
            cfg = ctx.tool_config
            input_dir_raw = cfg.get("input_dir", "{package_dir}/{build_type}/bin") if cfg else "{package_dir}/{build_type}/bin"
        input_dir = Path(formatter.resolve(str(input_dir_raw)))
        if not input_dir.is_absolute():
            input_dir = ctx.workspace_root / input_dir

        # Resolve output directory
        output_dir_raw = args.get("output_dir")
        if not output_dir_raw:
            cfg = ctx.tool_config
            output_dir_raw = cfg.get("output_dir", "{workspace_root}/_deploy") if cfg else "{workspace_root}/_deploy"
        output_dir = Path(formatter.resolve(str(output_dir_raw)))
        if not output_dir.is_absolute():
            output_dir = ctx.workspace_root / output_dir

        dry_run = bool(args.get("dry_run"))

        with log_section("Discovering apps"):
            logger.info(f"Input: {input_dir}")
            apps = _discover_apps(input_dir)
            if not apps:
                logger.error(f"No .html apps found in: {input_dir}")
                logger.info("Build and package the project first:")
                logger.info("  ./repo build --platform emscripten --build-type Release")
                logger.info("  ./repo package --platform emscripten --build-type Release")
                raise SystemExit(1)
            for app in apps:
                logger.info(f"  {app}")
            logger.info(f"Found {len(apps)} app(s)")

        # Clean output directory
        if output_dir.exists():
            if dry_run:
                logger.info(f"Would clean: {output_dir}")
            else:
                remove_tree_with_retries(output_dir)

        # Locate static assets (e.g. coi-serviceworker.js) from web/ dir
        web_dir = ctx.workspace_root / "web"

        with log_section("Publishing"):
            total_files = 0
            for stem in apps:
                app_dir = output_dir / stem
                suffixes = [".html", ".js", ".wasm"]
                copied = 0
                for suffix in suffixes:
                    src = input_dir / f"{stem}{suffix}"
                    if not src.exists():
                        continue
                    # Rename .html to index.html, keep others as-is
                    dest_name = "index.html" if suffix == ".html" else src.name
                    dest = app_dir / dest_name
                    if dry_run:
                        logger.info(f"  {src} -> {dest}")
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                    copied += 1

                # Copy static web assets (service workers, etc.) into each app dir
                for asset in web_dir.glob("*.js"):
                    if asset.name.startswith("_"):
                        continue
                    dest = app_dir / asset.name
                    if dry_run:
                        logger.info(f"  {asset} -> {dest}")
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(asset, dest)
                    copied += 1

                total_files += copied
                logger.info(f"  {stem}: {copied} file(s)")

            # Generate root index.html
            index_path = output_dir / "index.html"
            if dry_run:
                logger.info(f"  Would generate: {index_path}")
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                index_path.write_text(_generate_index(apps), encoding="utf-8")
            total_files += 1

            # Create .nojekyll
            nojekyll = output_dir / ".nojekyll"
            if dry_run:
                logger.info(f"  Would create: {nojekyll}")
            else:
                nojekyll.touch()
            total_files += 1

        if dry_run:
            logger.info(f"Dry run: {total_files} file(s) would be published to {output_dir}")
        else:
            logger.info(f"Published {total_files} file(s) to {output_dir}")
