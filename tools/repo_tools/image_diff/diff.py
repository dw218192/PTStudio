"""``image-diff`` tool: compare renderer captures against committed GT via FLIP.

Uses the ``flip_evaluator.evaluate`` Python API directly (no subprocess).
The primary metric is the max over per-tile mean FLIP -- whole-image mean
drowns localized regressions (a wrong shadow in a dark scene) in the
error-free background.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    ToolContext,
    log_section,
    logger,
    to_cmake_build_type,
)

from .common import (
    Case,
    ImageDiffConfig,
    build_editor_args,
    load_image_diff_config,
    require_flip_evaluator,
    run_launch,
    select_cases,
)


@dataclass
class WorstTile:
    """Bbox of the worst-scoring tile, in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    mean: float

    def as_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "mean": self.mean}


@dataclass
class CaseResult:
    """Outcome of a single diff case."""

    name: str
    tile_size: int
    score: float
    mean_flip: float
    threshold: float
    passed: bool
    worst_tile: WorstTile
    capture: Path
    heatmap: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": "max_tile_mean",
            "tile_size": self.tile_size,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "worst_tile": self.worst_tile.as_dict(),
            "mean_flip": self.mean_flip,
            "capture": str(self.capture),
            "heatmap": str(self.heatmap),
        }


def _compute_tile_max(err: Any, tile_size: int) -> tuple[float, WorstTile]:
    """Return ``(max_tile_mean, worst_tile)`` over tiles of *err*.

    *err* is the raw FLIP error map as an ``(H, W)`` float32 numpy array.
    Tiles at the right/bottom edge may be smaller than *tile_size* when the
    image dims aren't a multiple -- they contribute their actual area to
    the mean (no padding, no dropping).
    """
    H, W = err.shape
    best_mean = -1.0
    worst = WorstTile(x=0, y=0, w=0, h=0, mean=0.0)
    for y0 in range(0, H, tile_size):
        y1 = min(y0 + tile_size, H)
        for x0 in range(0, W, tile_size):
            x1 = min(x0 + tile_size, W)
            tm = float(err[y0:y1, x0:x1].mean())
            if tm > best_mean:
                best_mean = tm
                worst = WorstTile(
                    x=x0, y=y0, w=x1 - x0, h=y1 - y0, mean=tm,
                )
    return best_mean, worst


def _write_heatmap(
    heatmap_rgb: Any, worst: WorstTile, out_path: Path,
) -> None:
    """Save *heatmap_rgb* (H, W, 3) float32 to PNG with the worst-tile bbox.

    A 2 px cyan rectangle is drawn as an outline so the hotspot is visible
    against magma's warm palette.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    # FLIP's magma colormap returns floats in [0, 1]; quantize to 8-bit.
    rgb8 = np.clip(heatmap_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(rgb8, mode="RGB")
    draw = ImageDraw.Draw(img)
    x0, y0 = worst.x, worst.y
    x1, y1 = worst.x + worst.w - 1, worst.y + worst.h - 1
    # Two concentric 1-px rectangles -> 2 px visual thickness.
    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 255), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _run_case(
    case: Case,
    cfg: ImageDiffConfig,
    workspace_root: Path,
    build_type: str,
    logs_dir: Path,
    from_package: bool,
) -> CaseResult:
    """Capture, diff, and return the result for a single case."""
    if not case.gt.exists():
        raise FileNotFoundError(
            f"image-diff: ground truth missing for case '{case.name}': "
            f"{case.gt}\n  Bake it with: ./repo bake-gt --case {case.name}"
        )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    capture = cfg.out_dir / f"{case.name}.png"
    # Remove any stale capture so a failed editor run can't masquerade as
    # a pass from a previous run.
    if capture.exists():
        capture.unlink()

    launch_args = build_editor_args(
        case,
        capture_path=capture,
        renderer=case.renderer,
        frames=1,
        from_package=from_package,
    )
    run_launch(
        workspace_root,
        launch_args,
        build_type=build_type,
        log_file=logs_dir / f"image_diff_capture_{case.name}.log",
    )
    if not capture.exists():
        raise RuntimeError(
            f"image-diff: capture was not produced for case '{case.name}' "
            f"(expected {capture})"
        )

    flip = require_flip_evaluator()
    # Raw error map for tile scoring.
    err_raw, mean_flip, _ = flip.evaluate(
        str(case.gt), str(capture), "LDR", applyMagma=False,
    )
    # err_raw shape: (H, W, 1) float32, values in [0, 1].
    err2d = err_raw[..., 0]
    score, worst = _compute_tile_max(err2d, cfg.tile_size)

    # Colorized heatmap for debug output. Separate call -- flip_evaluator's
    # API doesn't expose both outputs in a single invocation, and the cost
    # is negligible (one FLIP pass per case).
    heatmap_rgb, _, _ = flip.evaluate(
        str(case.gt), str(capture), "LDR", applyMagma=True, computeMeanError=False,
    )
    heatmap = cfg.out_dir / f"{case.name}.diff.png"
    _write_heatmap(heatmap_rgb, worst, heatmap)

    return CaseResult(
        name=case.name,
        tile_size=cfg.tile_size,
        score=score,
        mean_flip=float(mean_flip),
        threshold=case.threshold,
        passed=score <= case.threshold,
        worst_tile=worst,
        capture=capture,
        heatmap=heatmap,
    )


def _write_summary(
    summary_path: Path, results: list[CaseResult], tile_size: int,
) -> None:
    summary = {
        "tile_size": tile_size,
        "cases": [r.as_dict() for r in results],
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )


def _print_results(results: list[CaseResult]) -> None:
    logger.info("image-diff summary (metric: max tile-mean FLIP):")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = (
            f"  {status:4s}  {r.name:24s}  score={r.score:.5f}  "
            f"threshold={r.threshold:.5f}  mean={r.mean_flip:.5f}"
        )
        if not r.passed:
            w = r.worst_tile
            line += f"  worst=({w.x},{w.y},{w.w}x{w.h})"
        logger.info(line)


class ImageDiffTool(RepoTool):
    name = "image-diff"
    help = "Diff renderer captures against golden GT via FLIP"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "--case",
            "case_name",
            type=str,
            default=None,
            help="Run only the named case (default: all cases)",
        )(cmd)
        cmd = click.option(
            "-c", "--config",
            type=click.Choice(
                ["debug", "release", "relwithdebinfo", "minsizerel"],
                case_sensitive=False,
            ),
            default=None,
            help="Build configuration (overrides --build-type)",
        )(cmd)
        cmd = click.option(
            "--from-package",
            is_flag=True,
            default=None,
            help="Capture from packaged artifacts instead of build dir (CI)",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {"case_name": None, "config": None, "from_package": False}

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        # Fail loud if the dep is missing, up front rather than mid-loop.
        require_flip_evaluator()
        cfg = load_image_diff_config(ctx.workspace_root, ctx.config)

        case_name = args.get("case_name")
        cases = select_cases(cfg, case_name)

        build_type_override = args.get("config")
        if build_type_override:
            build_type = to_cmake_build_type(build_type_override)
        else:
            build_type = ctx.dimensions.get("build_type", "Debug")

        logs_dir = Path(ctx.tokens["logs_root"])
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

        from_package = bool(args.get("from_package"))
        results: list[CaseResult] = []
        for case in cases:
            with log_section(f"image-diff: {case.name}"):
                results.append(_run_case(
                    case, cfg, ctx.workspace_root, build_type, logs_dir,
                    from_package,
                ))

        summary_path = cfg.out_dir / "summary.json"
        _write_summary(summary_path, results, cfg.tile_size)
        _print_results(results)
        logger.info(f"Summary written to {summary_path}")

        failed = [r for r in results if not r.passed]
        if failed:
            logger.error(
                f"image-diff: {len(failed)} of {len(results)} case(s) exceeded "
                f"threshold"
            )
            sys.exit(1)
