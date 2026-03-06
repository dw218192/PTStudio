"""Slang reflection metadata generation command."""

import sys
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    ShellCommand,
    ToolContext,
    logger,
    resolve_path,
)


class SlangReflectTool(RepoTool):
    name = "slang_reflect"
    help = "Generate C++ headers from Slang shader reflection"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "-f",
            "--force",
            is_flag=True,
            default=None,
            help="Regenerate headers even if outputs are up to date",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "force": False,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        """Generate C++ shader metadata headers via slang_reflect binary."""
        root = ctx.workspace_root
        tokens = ctx.tokens
        force = args.get("force", False)

        shaders = args.get("shaders")
        if shaders is None:
            shaders = ctx.config.get("slang_reflect", {}).get("shaders", [])
        if not shaders:
            logger.warning("No slang_reflect shaders configured.")
            return

        conanbuild = Path(tokens["build_dir"]) / "conanbuild"
        logs_dir = Path(tokens["logs_root"])
        logs_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        skipped = 0

        for shader in shaders:
            input_value = shader.get("input")
            if not input_value:
                continue

            input_path = resolve_path(root, str(input_value), tokens)
            output_path = resolve_path(root, str(shader["output"]), tokens)
            namespace = shader.get("namespace", "shader_metadata")
            entries = shader.get("entries", [])

            if not input_path.exists():
                logger.error(f"Shader input not found: {input_path}")
                sys.exit(1)

            if not entries:
                logger.error(f"No entry points specified for: {input_path}")
                sys.exit(1)

            # mtime-based skip logic
            if (
                not force
                and output_path.exists()
                and output_path.stat().st_mtime >= input_path.stat().st_mtime
            ):
                logger.info(f"Skipping up-to-date: {output_path}")
                skipped += 1
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"slang_reflect_{input_path.stem}.log"

            cmd = [
                "slang_reflect",
                str(input_path),
                "-o", str(output_path),
                "-n", namespace,
            ]
            for entry in entries:
                cmd.extend(["-e", entry])

            ShellCommand(cmd, env_script=conanbuild).exec(log_file=log_file)
            logger.info(f"slang_reflect generated: {output_path}")
            generated += 1

        logger.info(f"slang_reflect generated {generated} header(s)")
        if skipped:
            logger.info(f"slang_reflect skipped {skipped} up-to-date header(s)")
