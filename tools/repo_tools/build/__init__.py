"""Build subcommand implementation (PTStudio project override)."""

from __future__ import annotations

from typing import Any

import click

from repo_tools.core import McpLogRecord, RepoTool, ToolContext

from .command import build_command


class BuildTool(RepoTool):
    name = "build"
    help = "Build the project"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "-x", "--rebuild",
            is_flag=True,
            default=None,
            help=(
                "Rebuild flag: removes build configuration folder before building "
                "(Use clean tool for a full clean of build and dependencies folders)"
            ),
        )(cmd)
        cmd = click.option(
            "-u", "--update-lock",
            is_flag=True,
            default=None,
            help="Update lock flag: forces regeneration of conan.lock",
        )(cmd)
        cmd = click.option(
            "-c", "--configure-only",
            is_flag=True,
            default=None,
            help=(
                "Configure only flag: runs conan install and cmake configure, "
                "but skips building"
            ),
        )(cmd)
        cmd = click.option(
            "-b", "--build-only",
            is_flag=True,
            default=None,
            help=(
                "Build only flag: skips conan install and cmake configure, "
                "only runs build"
            ),
        )(cmd)
        cmd = click.option(
            "--conan-profile",
            default=None,
            help="Conan profile (default: default)",
        )(cmd)
        cmd = click.option(
            "--windowing",
            type=click.Choice(["glfw", "null"]),
            default=None,
            help="Windowing backend (default: glfw)",
        )(cmd)
        cmd = click.option(
            "--host-tools-only",
            is_flag=True,
            default=None,
            help=(
                "Build only host tools (e.g. usdz_pack) via their own Conan "
                "packages and run their prebuild steps. Skips the main app "
                "build. Not valid with --platform emscripten."
            ),
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "rebuild": False,
            "update_lock": False,
            "configure_only": False,
            "build_only": False,
            "conan_profile": "default",
            "windowing": "glfw",
            "host_tools_only": False,
            "prebuild": {},
            "postbuild": {},
            "conan": {},
        }

    # Path substrings identifying third-party build output (Conan dep cache,
    # framework venv, framework source). Warnings emitted from these paths
    # are suppressed in MCP output -- project-local warnings still come
    # through. Both forward- and back-slash forms covered by testing .conan2
    # and the framework anchor names, which appear in either layout.
    _THIRD_PARTY_WARNING_MARKERS = (
        ".conan2",
        "_managed",
        "tools/framework",
        "tools\\framework",
    )

    def format_mcp_output(
        self, records: list[McpLogRecord], returncode: int
    ) -> str | None:
        """Show prebuild results + build outcome, skip Conan/CMake noise."""
        lines: list[str] = []
        for r in records:
            msg = r.message
            lower = msg.lower()
            is_warning = "warning" in lower or "error" in lower
            if is_warning and any(m in msg for m in self._THIRD_PARTY_WARNING_MARKERS):
                continue
            if r.level in ("error", "critical", "warning"):
                lines.append(msg)
            elif any(k in msg for k in ("[OK]", "[FAIL]", "CMake build", "FAILED")):
                lines.append(msg)
            elif r.level == "output" and is_warning:
                lines.append(msg)
        if not lines:
            lines.append("Build completed successfully")
        lines.append("\nFull log: _build/logs/mcp/build.log")
        return "\n".join(lines)

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        build_command(ctx, args, self.name)
