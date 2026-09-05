"""``bake-gt`` tool: bake path-traced ground-truth PNGs for image-diff cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from tasks.image_diff_support import (
    Case,
    ImageDiffConfig,
    build_editor_args,
    load_image_diff_config,
    run_launch,
    select_cases,
)
from tasks.utils import (
    ProjectContext,
    log_section,
    logger,
    to_cmake_build_type,
)
from tasks.utils.project import load_context, project_options


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
        raise RuntimeError(f"bake-gt: editor did not produce {case.gt} for case '{case.name}'")
    return case.gt


def run(ctx: ProjectContext, args: dict[str, Any]) -> None:
    args = ctx.arguments("bake-gt", {"case_name": None, "config": None}, args)
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
                case,
                cfg,
                ctx.workspace_root,
                build_type,
                logs_dir,
            )
            logger.info(f"Wrote GT: {path}")


@click.command(name="bake-gt", help="Bake path-traced ground-truth PNGs for image-diff cases")
@project_options
@click.option(
    "--case",
    "case_name",
    type=str,
    default=None,
    help="Bake only the named case (default: all cases)",
)
@click.option(
    "-c",
    "--config",
    type=click.Choice(
        ["debug", "release", "relwithdebinfo", "minsizerel"],
        case_sensitive=False,
    ),
    default=None,
    help="Build configuration (overrides --build-type)",
)
@click.pass_context
def main(cli: click.Context, platform: str, build_type: str, **args: Any) -> None:
    context = load_context(platform, args.get("config") or build_type, passthrough=cli.args)
    run(context, args)


if __name__ == "__main__":
    main()
