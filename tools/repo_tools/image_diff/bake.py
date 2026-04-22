"""``bake-gt`` tool: bake path-traced ground-truth PNGs for image-diff cases."""

from __future__ import annotations

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
    run_launch,
    select_cases,
)


def _bake_case(
    case: Case,
    cfg: ImageDiffConfig,
    workspace_root: Path,
    build_type: str,
    logs_dir: Path,
) -> Path:
    """Render *case* with the GT renderer/frames and write to ``case.gt``.

    Returns the written path so callers can log it.
    """
    case.gt.parent.mkdir(parents=True, exist_ok=True)
    launch_args = build_editor_args(
        case,
        capture_path=case.gt,
        renderer=cfg.gt_bake.renderer,
        frames=cfg.gt_bake.frames,
    )
    run_launch(
        workspace_root,
        launch_args,
        build_type=build_type,
        log_file=logs_dir / f"bake_gt_{case.name}.log",
    )
    if not case.gt.exists():
        raise RuntimeError(
            f"bake-gt: editor did not produce {case.gt} for case '{case.name}'"
        )
    return case.gt


class BakeGtTool(RepoTool):
    name = "bake-gt"
    help = "Bake path-traced ground-truth PNGs for image-diff cases"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "--case",
            "case_name",
            type=str,
            default=None,
            help="Bake only the named case (default: all cases)",
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
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {"case_name": None, "config": None}

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        cfg = load_image_diff_config(ctx.workspace_root, ctx.config)
        cases = select_cases(cfg, args.get("case_name"))

        build_type_override = args.get("config")
        if build_type_override:
            build_type = to_cmake_build_type(build_type_override)
        else:
            build_type = ctx.dimensions.get("build_type", "Debug")

        logs_dir = Path(ctx.tokens["logs_root"])

        for case in cases:
            with log_section(f"bake-gt: {case.name}"):
                path = _bake_case(
                    case, cfg, ctx.workspace_root, build_type, logs_dir,
                )
                logger.info(f"Wrote GT: {path}")
