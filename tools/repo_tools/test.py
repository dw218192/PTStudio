"""Test subcommand -- discovers and runs test executables."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    McpLogRecord,
    RepoTool,
    ToolContext,
    logger,
    to_cmake_build_type,
)
from repo_tools.launch import _can_run, _run_tests


class TestTool(RepoTool):
    name = "test"
    help = "Run test executables"

    def setup(self, cmd: click.Command) -> click.Command:
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
            "-v", "--verbose",
            is_flag=True,
            default=None,
            help="Verbose test output",
        )(cmd)
        cmd = click.option(
            "--from-package",
            is_flag=True,
            default=None,
            help="Run from packaged artifacts instead of build dir (CI)",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "config": None,
            "verbose": False,
            "from_package": False,
        }

    def format_mcp_output(
        self, records: list[McpLogRecord], returncode: int
    ) -> str | None:
        """Show only test summary and failures, not per-test GPU logs."""
        lines: list[str] = []
        for r in records:
            msg = r.message
            if any(
                k in msg
                for k in ("PASSED:", "FAILED:", "Test summary", "Total:", "Passed:", "Failed:", "All tests passed")
            ):
                lines.append(msg)
            elif r.level in ("error", "critical", "warning"):
                lines.append(msg)
        lines.append("\nFull log: _build/logs/mcp/test.log")
        return "\n".join(lines)

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
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

        sys.exit(_run_tests(context, bool(args.get("verbose")),
                            from_package=bool(args.get("from_package"))))
