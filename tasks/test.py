"""Test task -- discovers and runs test executables."""

from __future__ import annotations

import sys
from typing import Any

import click

from tasks import image_diff
from tasks.launch import _can_run, _run_tests
from tasks.utils import (
    ProjectContext,
    log_section,
    logger,
    to_cmake_build_type,
)
from tasks.utils.project import load_context, project_options


def run(ctx: ProjectContext, args: dict[str, Any]) -> None:
    args = ctx.arguments("test", {"config": None, "verbose": False, "from_package": False}, args)
    config_val = args.get("config")
    if config_val:
        build_type = to_cmake_build_type(config_val)
    else:
        build_type = ctx.dimensions.get("build_type", "Debug")

    platform_id = ctx.dimensions.get("platform", "")

    context: dict[str, Any] = {
        "workspace_root": str(ctx.workspace_root),
        "build_dir": ctx.tokens["build_dir"],
        "conan_deps_root": ctx.tokens["conan_deps_root"],
        "package_dir": ctx.tokens["package_dir"],
        "platform": platform_id,
        "build_type": build_type,
        "logs_root": ctx.tokens["logs_root"],
    }

    if not _can_run(context):
        if platform_id == "emscripten":
            logger.error("emsdk not found. Build with --platform emscripten first.")
        else:
            logger.error(f"Cannot run {platform_id} binaries on this host")
        sys.exit(1)

    tests_rc = _run_tests(
        context, bool(args.get("verbose")), from_package=bool(args.get("from_package"))
    )

    # Image-diff is a native-only post-step. Emscripten uses the browser
    # to run tests and has no host-side PNG capture; skip there.
    # Also skip when running from a package (no committed GT next to
    # the built artifacts) and when core tests already failed.
    run_image_diff = (
        tests_rc == 0 and platform_id != "emscripten" and not bool(args.get("from_package"))
    )
    if run_image_diff:
        with log_section("image-diff"):
            try:
                image_diff.run(ctx, {})
            except SystemExit as exc:
                if exc.code:
                    tests_rc = int(exc.code) if isinstance(exc.code, int) else 1

    sys.exit(tests_rc)


@click.command(name="test", help="Run test executables")
@project_options
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
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=None,
    help="Verbose test output",
)
@click.option(
    "--from-package",
    is_flag=True,
    default=None,
    help="Run from packaged artifacts instead of build dir (CI)",
)
@click.pass_context
def main(cli: click.Context, platform: str, build_type: str, **args: Any) -> None:
    context = load_context(platform, args.get("config") or build_type, passthrough=cli.args)
    run(context, args)


if __name__ == "__main__":
    main()
