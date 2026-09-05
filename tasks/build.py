"""Build task."""

from __future__ import annotations

from typing import Any

import click

from tasks.build_support.command import build_command
from tasks.utils import ProjectContext
from tasks.utils.project import load_context, project_options


def run(ctx: ProjectContext, args: dict[str, Any]) -> None:
    args = ctx.arguments(
        "build",
        {
            "rebuild": False,
            "update_lock": False,
            "configure_only": False,
            "build_only": False,
            "conan_profile": "default",
            "windowing": "glfw",
            "host_tools_only": False,
            "conan": {},
        },
        args,
    )
    build_command(ctx, args)


@click.command(name="build", help="Build the project")
@project_options
@click.option(
    "-x",
    "--rebuild",
    is_flag=True,
    default=None,
    help=(
        "Rebuild flag: removes build configuration folder before building "
        "(Use clean tool for a full clean of build and dependencies folders)"
    ),
)
@click.option(
    "-u",
    "--update-lock",
    is_flag=True,
    default=None,
    help="Update lock flag: forces regeneration of conan.lock",
)
@click.option(
    "-c",
    "--configure-only",
    is_flag=True,
    default=None,
    help=("Configure only flag: runs conan install and cmake configure, but skips building"),
)
@click.option(
    "-b",
    "--build-only",
    is_flag=True,
    default=None,
    help=("Build only flag: skips conan install and cmake configure, only runs build"),
)
@click.option(
    "--conan-profile",
    default=None,
    help="Conan profile (default: default)",
)
@click.option(
    "--windowing",
    type=click.Choice(["glfw", "null"]),
    default=None,
    help="Windowing backend (default: glfw)",
)
@click.option(
    "--host-tools-only",
    is_flag=True,
    default=None,
    help=(
        "Build only host tools (e.g. usdz_pack) via their own Conan "
        "packages and run their prebuild steps. Skips the main app "
        "build. Not valid with --platform emscripten."
    ),
)
@click.pass_context
def main(cli: click.Context, platform: str, build_type: str, **args: Any) -> None:
    context = load_context(platform, args.get("config") or build_type, passthrough=cli.args)
    run(context, args)


if __name__ == "__main__":
    main()
