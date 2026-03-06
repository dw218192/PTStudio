"""Slang shader compilation command."""

import sys
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    ShellCommand,
    ToolContext,
    glob_paths,
    logger,
    resolve_path,
)



# ── Shader resolution ───────────────────────────────────────────────


def _resolve_slang_shaders(
    root: Path, config: dict, tokens: dict[str, str], args: dict[str, Any]
) -> tuple[list[tuple[Path, Path, bool]], int]:
    """Resolve shader entries, returning (input, output, reflect) tuples."""
    shaders = args.get("shaders")
    if shaders is None:
        shaders = config.get("slangc", {}).get("shaders", [])

    if not shaders:
        return [], 0
    if not isinstance(shaders, list):
        logger.warning("Slang shader configuration must be a list.")
        return [], 0

    resolved: list[tuple[Path, Path, bool]] = []
    errors = 0
    seen_outputs: set[Path] = set()

    for idx, shader in enumerate(shaders):
        if not isinstance(shader, dict):
            logger.warning(
                f"Skipping invalid shader entry at index {idx}: "
                f"expected dict, got {type(shader).__name__} ({shader!r})"
            )
            continue
        input_value = shader.get("input")
        if not input_value:
            continue
        output_value = shader.get("output")
        reflect = bool(shader.get("reflect", False))

        input_pattern = resolve_path(root, str(input_value), tokens)
        input_paths = [
            path for path in glob_paths(input_pattern) if path.is_file()
        ]
        if not input_paths:
            logger.error(f"No shader inputs matched: {input_pattern}")
            errors += 1
            continue

        output_pattern_text = None
        if output_value:
            output_pattern_text = str(resolve_path(root, str(output_value), tokens))
            if "*" not in output_pattern_text and len(input_paths) > 1:
                logger.error(
                    "Output path must include '*' when multiple inputs match: "
                    f"{output_pattern_text}"
                )
                errors += 1
                continue

        for input_path in input_paths:
            if output_value:
                output_text = output_pattern_text
                if "*" in output_pattern_text:
                    output_text = output_pattern_text.replace("*", input_path.stem)
                output_path = Path(output_text)
            else:
                output_path = input_path.with_suffix(".wgsl")

            if output_path in seen_outputs:
                logger.error(f"Duplicate shader output path: {output_path}")
                errors += 1
                continue
            seen_outputs.add(output_path)
            resolved.append((input_path, output_path, reflect))

    return resolved, errors


def _should_compile_shader(input_path: Path, output_path: Path, force: bool) -> bool:
    if force:
        return True
    if not output_path.exists():
        return True
    return output_path.stat().st_mtime < input_path.stat().st_mtime


def _emit_reflection_json(
    compiler: str,
    input_path: Path,
    output_path: Path,
    conanbuild: Path,
    passthrough_args: list[str],
) -> None:
    """Emit reflection JSON via slangc -reflection-json."""
    reflect_path = output_path.with_suffix(".reflect.json")
    reflect_path.parent.mkdir(parents=True, exist_ok=True)

    reflect_cmd = [
        compiler,
        str(input_path),
        "-target", "wgsl",
        "-reflection-json", str(reflect_path),
    ]
    reflect_cmd.extend(passthrough_args)

    logs_dir = reflect_path.parent
    log_file = logs_dir / f"slangc_reflect_{input_path.stem}.log"
    ShellCommand(reflect_cmd, env_script=conanbuild).exec(log_file=log_file)
    logger.info(f"slangc emitted reflection JSON: {reflect_path}")


class SlangcTool(RepoTool):
    name = "slangc"
    help = "Compile Slang shaders"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "-f",
            "--force",
            is_flag=True,
            default=None,
            help="Recompile shaders even if outputs are up to date",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "force": False,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        """Compile Slang shaders configured in config.yaml."""
        root = ctx.workspace_root
        config = ctx.config
        tokens = ctx.tokens

        # Explicit compiler path override from args or config
        compiler_path = args.get("compiler_path")
        if compiler_path is None:
            compiler_path = config.get("slangc", {}).get("compiler_path")
        if compiler_path:
            compiler = str(resolve_path(root, str(compiler_path), tokens))
        else:
            compiler = "slangc"

        conanbuild = Path(tokens["build_dir"]) / "conanbuild"

        shaders, errors = _resolve_slang_shaders(root, config, tokens, args)
        if errors:
            sys.exit(1)
        if not shaders:
            logger.warning("No Slang shaders configured.")
            return

        logs_dir = Path(tokens["logs_root"])
        logs_dir.mkdir(parents=True, exist_ok=True)

        compiled = 0
        skipped = 0
        for input_path, output_path, reflect in shaders:
            if not input_path.exists():
                logger.error(f"Shader input not found: {input_path}")
                sys.exit(1)

            if _should_compile_shader(input_path, output_path, args["force"]):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = logs_dir / f"slangc_{output_path.stem}.log"
                cmd = [
                    compiler,
                    str(input_path),
                    "-o",
                    str(output_path),
                    "-target",
                    "wgsl",
                ]
                cmd.extend(ctx.passthrough_args)
                ShellCommand(cmd, env_script=conanbuild).exec(log_file=log_file)
                compiled += 1
            else:
                logger.info(f"Skipping up-to-date shader: {input_path}")
                skipped += 1

            # Emit reflection JSON sidecar if requested (even if WGSL was up-to-date)
            if reflect:
                reflect_path = output_path.with_suffix(".reflect.json")
                needs_reflect = (
                    args["force"]
                    or not reflect_path.exists()
                    or reflect_path.stat().st_mtime < output_path.stat().st_mtime
                )
                if needs_reflect:
                    _emit_reflection_json(
                        compiler, input_path, output_path,
                        conanbuild, ctx.passthrough_args,
                    )

        logger.info(f"slangc compiled {compiled} shader(s)")
        if skipped:
            logger.info(f"slangc skipped {skipped} up-to-date shader(s)")
