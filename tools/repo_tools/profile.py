"""Profile tool — download and launch Tracy profiler viewer."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import click

from repo_tools.core import RepoTool, ToolContext, invoke_tool, logger

# Tracy release to download — update when upgrading tracy Conan package
_TRACY_VERSION = "0.11.1"
_TRACY_BASE_URL = (
    f"https://github.com/wolfpld/tracy/releases/download/v{_TRACY_VERSION}"
)

_PLATFORM_ASSETS = {
    "win32": {
        "archive": f"Tracy-{_TRACY_VERSION}-win64.zip",
        "binary_glob": "Tracy.exe",
        "binary_name": "Tracy.exe",
    },
    "linux": {
        "archive": f"Tracy-{_TRACY_VERSION}-linux-x64.zip",
        "binary_glob": "Tracy-*",
        "binary_name": "tracy",
    },
    "darwin": {
        "archive": f"Tracy-{_TRACY_VERSION}-macos-universal.zip",
        "binary_glob": "Tracy.app",
        "binary_name": "Tracy.app",
    },
}


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

        launch_cmd = _viewer_launch_cmd(viewer)
        viewer_proc = subprocess.Popen(launch_cmd)
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


def _host_platform() -> str:
    """Return sys.platform normalized to win32/linux/darwin."""
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform  # win32 or darwin


def _viewer_launch_cmd(viewer: Path) -> list[str]:
    """Return the command to launch the viewer, handling macOS .app bundles."""
    if sys.platform == "darwin" and viewer.suffix == ".app":
        return ["open", "-a", str(viewer)]
    return [str(viewer)]


def _ensure_tracy_viewer(workspace_root: Path) -> Path:
    """Download Tracy profiler viewer if not cached."""
    platform = _host_platform()
    asset = _PLATFORM_ASSETS.get(platform)
    if not asset:
        raise RuntimeError(f"Tracy profiler viewer not available for platform: {platform}")

    cache_dir = workspace_root / "_build" / "tools" / "tracy"
    viewer = cache_dir / asset["binary_name"]

    if viewer.exists():
        return viewer

    url = f"{_TRACY_BASE_URL}/{asset['archive']}"
    logger.info(f"Downloading Tracy profiler v{_TRACY_VERSION} from {url}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    resp = urlopen(url)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        zf.extractall(cache_dir)

    candidates = list(cache_dir.rglob(asset["binary_glob"]))
    if not candidates:
        raise RuntimeError(
            f"{asset['binary_name']} not found in downloaded archive from {url}"
        )

    # Move to expected location if nested in a subfolder
    found = candidates[0]
    if found != viewer:
        found.rename(viewer)

    # Make executable on Unix
    if platform != "win32" and viewer.is_file():
        viewer.chmod(viewer.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    logger.info(f"Tracy viewer cached at {viewer}")
    return viewer
