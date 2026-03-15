"""Profile tool — download and launch Tracy profiler viewer."""

from __future__ import annotations

import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import click

from repo_tools.core import RepoTool, ToolContext, invoke_tool, logger

# Tracy release to download — update when upgrading tracy Conan package
_TRACY_VERSION = "0.11.1"
_TRACY_RELEASE_URL = (
    f"https://github.com/wolfpld/tracy/releases/download/v{_TRACY_VERSION}/"
    f"Tracy-{_TRACY_VERSION}-win64.zip"
)


class ProfileTool(RepoTool):
    name = "profile"
    help = "Launch Tracy profiler viewer alongside the editor"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "--viewer-only",
            is_flag=True,
            default=False,
            help="Launch only the Tracy viewer without starting the editor",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {"viewer_only": False}

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        viewer = _ensure_tracy_viewer(ctx.workspace_root)
        logger.info(f"Tracy viewer: {viewer}")

        viewer_proc = subprocess.Popen([str(viewer)])
        logger.info("Tracy viewer started — waiting for profiled application to connect")

        if args.get("viewer_only"):
            try:
                viewer_proc.wait()
            except KeyboardInterrupt:
                viewer_proc.terminate()
            return

        try:
            invoke_tool("launch", ctx.tokens, ctx.config, dimensions=ctx.dimensions)
        except KeyboardInterrupt:
            pass
        finally:
            if viewer_proc.poll() is None:
                logger.info("Stopping Tracy viewer")
                viewer_proc.terminate()


def _ensure_tracy_viewer(workspace_root: Path) -> Path:
    """Download Tracy profiler viewer if not cached."""
    cache_dir = workspace_root / "_build" / "tools" / "tracy"
    viewer = cache_dir / "Tracy.exe"  # Windows only for now

    if viewer.exists():
        return viewer

    logger.info(f"Downloading Tracy profiler v{_TRACY_VERSION}...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    resp = urlopen(_TRACY_RELEASE_URL)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        zf.extractall(cache_dir)

    # Find the exe — may be nested in a subfolder
    candidates = list(cache_dir.rglob("Tracy.exe"))
    if not candidates:
        raise RuntimeError(
            f"Tracy.exe not found in downloaded archive from {_TRACY_RELEASE_URL}"
        )

    # Move to expected location if nested
    if candidates[0] != viewer:
        candidates[0].rename(viewer)

    logger.info(f"Tracy viewer cached at {viewer}")
    return viewer
