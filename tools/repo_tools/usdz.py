"""USDZ packaging tool — builds and runs usdz_pack (two-phase build).

Phase 1: cmake --build --target usdz_pack (compiles the host tool)
Phase 2: run usdz_pack for each scene (creates .usdz from .usda)
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

from repo_tools.core import RepoTool, ShellCommand, ToolContext, logger


def _newest_mtime(paths: list[Path]) -> float:
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0)


def _collect_deps(input_path: Path) -> list[Path]:
    """Collect .usda source and any files it references via @...@ asset paths."""
    deps = [input_path]
    try:
        text = input_path.read_text(encoding="utf-8")
        parent = input_path.parent
        for line in text.split("\n"):
            start = line.find("@")
            while start >= 0:
                end = line.find("@", start + 1)
                if end < 0:
                    break
                ref = line[start + 1 : end]
                ref_path = (parent / ref).resolve()
                if ref_path.exists():
                    deps.append(ref_path)
                start = line.find("@", end + 1)
    except Exception as e:
        logger.warning(f"Failed to parse dependencies from {input_path}: {e}")
    return deps


class UsdzTool(RepoTool):
    name = "usdz"
    help = "Package USDA scenes as USDZ archives via usdz_pack"

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        platform = ctx.dimensions.get("platform", "")
        if platform == "emscripten":
            logger.info("Skipping usdz packaging (host-only tool, not available on Emscripten)")
            return

        root = ctx.workspace_root
        scenes = ctx.config.get("usdz", {}).get("scenes", [])
        if not scenes:
            logger.info("No USDZ scenes configured.")
            return

        build_dir = Path(ctx.tokens["build_dir"])
        # usdz_pack needs runtime DLLs (USD), not just compiler tools
        conanrun = build_dir / "conanrun"
        logs_dir = Path(ctx.tokens["logs_root"])
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Check which scenes need updating
        work = []
        for entry in scenes:
            input_path = root / entry["input"]
            output_path = root / entry["output"]
            if not input_path.exists():
                logger.warning(f"USDZ input not found: {input_path}")
                continue
            deps = _collect_deps(input_path)
            if output_path.exists() and output_path.stat().st_mtime > _newest_mtime(deps):
                logger.info(f"Skipping up-to-date: {output_path}")
                continue
            work.append((input_path, output_path))

        if not work:
            logger.info("usdz packaged 0 scene(s)")
            return

        # Find the usdz_pack binary (built by host tools phase)
        exe_name = "usdz_pack.exe" if sys.platform == "win32" else "usdz_pack"
        usdz_pack = build_dir / "bin" / exe_name
        if not usdz_pack.exists():
            for candidate in build_dir.rglob(exe_name):
                if candidate.is_file() and candidate.stat().st_size > 0:
                    usdz_pack = candidate
                    break
        if not usdz_pack.exists():
            logger.error(f"usdz_pack binary not found in {build_dir}")
            logger.error("Run a native build first (usdz_pack is a host-only tool)")
            sys.exit(1)

        # Phase 2: run usdz_pack for each scene
        for input_path, output_path in work:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"usdz_{output_path.stem}.log"
            cmd = ShellCommand(
                [str(usdz_pack), str(input_path), str(output_path)],
                env_script=conanrun,
            )
            cmd.exec(log_file=log_file)
            size_kb = output_path.stat().st_size / 1024
            logger.info(f"Packaged {output_path.name} ({size_kb:.1f} KB)")

        logger.info(f"usdz packaged {len(work)} scene(s)")
