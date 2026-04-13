"""Slang shader compilation command."""

import re
import sys
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    McpLogRecord,
    RepoTool,
    ShellCommand,
    ToolContext,
    glob_paths,
    logger,
    resolve_path,
)



# ── Shader resolution ───────────────────────────────────────────────


def _insert_suffix(path: Path, suffix: str) -> Path:
    """Insert a filename suffix before the final extension."""
    if not suffix:
        return path
    return path.with_name(path.stem + suffix + path.suffix)


def _shader_variants(shader: dict) -> list[dict]:
    """Return the variant list for a shader entry.

    Each returned dict has at minimum ``defines`` (list) and ``suffix`` (str).
    If the entry has no explicit ``variants``, a single implicit variant is
    synthesised from the top-level ``defines``.
    """
    variants_cfg = shader.get("variants")
    if variants_cfg is None:
        return [{
            "defines": list(shader.get("defines", [])),
            "suffix": "",
        }]
    out: list[dict] = []
    for variant in variants_cfg:
        if not isinstance(variant, dict):
            raise ValueError(
                f"Invalid variant entry: expected dict, got {type(variant).__name__}"
            )
        out.append({
            "defines": list(variant.get("defines", [])),
            "suffix": str(variant.get("suffix", "")),
        })
    return out


def _resolve_slang_shaders(
    root: Path, config: dict, tokens: dict[str, str], args: dict[str, Any]
) -> tuple[list[tuple[Path, Path, bool, list[str]]], int]:
    """Resolve shader entries. One tuple per (input × variant).

    Each shader entry may declare a ``variants`` list. Each variant has
    ``defines`` (list[str]) and ``suffix`` (str). The suffix is inserted
    before the output filename's extension (e.g. ``forward.wgsl`` with
    suffix ``"_no_debug"`` -> ``forward_no_debug.wgsl``). When ``variants``
    is omitted, the entry is treated as a single base variant (suffix="",
    defines from the entry's top-level ``defines`` field).
    """
    shaders = args.get("shaders")
    if shaders is None:
        shaders = config.get("slangc", {}).get("shaders", [])

    if not shaders:
        return [], 0
    if not isinstance(shaders, list):
        logger.warning("Slang shader configuration must be a list.")
        return [], 0

    resolved: list[tuple[Path, Path, bool, list[str]]] = []
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

        try:
            variants = _shader_variants(shader)
        except ValueError as e:
            logger.error(f"Shader entry {idx} ({input_value}): {e}")
            errors += 1
            continue

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
                base_output = Path(output_text)
            else:
                base_output = input_path.with_suffix(".wgsl")

            for variant in variants:
                output_path = _insert_suffix(base_output, variant["suffix"])
                if output_path in seen_outputs:
                    logger.error(f"Duplicate shader output path: {output_path}")
                    errors += 1
                    continue
                seen_outputs.add(output_path)
                # Only the base (no-suffix) variant emits reflection JSON —
                # shader_codegen consumes it for define-agnostic C++ metadata.
                variant_reflect = reflect and not variant["suffix"]
                resolved.append((
                    input_path, output_path, variant_reflect, variant["defines"],
                ))

    return resolved, errors


def _should_compile_shader(
    input_path: Path,
    output_path: Path,
    force: bool,
    search_paths: list[Path] | None = None,
) -> bool:
    if force:
        return True
    if not output_path.exists():
        return True
    out_mtime = output_path.stat().st_mtime
    # Check input file and all .slang siblings (potential imports)
    for slang_file in input_path.parent.glob("*.slang"):
        if slang_file.stat().st_mtime > out_mtime:
            return True
    # Check search path directories for imported modules
    for sp in (search_paths or []):
        if sp.is_dir():
            for slang_file in sp.glob("*.slang"):
                if slang_file.stat().st_mtime > out_mtime:
                    return True
    return False


def _emit_reflection_json(
    compiler: str,
    input_path: Path,
    output_path: Path,
    conanbuild: Path,
    passthrough_args: list[str],
    search_paths: list[Path] | None = None,
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
    for sp in (search_paths or []):
        reflect_cmd.extend(["-I", str(sp)])
    reflect_cmd.extend(passthrough_args)

    logs_dir = reflect_path.parent
    log_file = logs_dir / f"slangc_reflect_{input_path.stem}.log"
    ShellCommand(reflect_cmd, env_script=conanbuild).exec(log_file=log_file)
    logger.info(f"slangc emitted reflection JSON: {reflect_path}")


# Slang's WGSL backend emits texture_2d_array<f32> for Texture2DArray<float>
# even when used with SampleCmpLevelZero / SamplerComparisonState.  WGSL requires
# texture_depth_2d_array for comparison sampling.  This mapping covers all depth
# texture shapes; only declarations whose variables appear in
# textureSampleCompareLevel calls are patched.
_DEPTH_TYPE_MAP = {
    "texture_2d<f32>": "texture_depth_2d",
    "texture_2d_array<f32>": "texture_depth_2d_array",
    "texture_cube<f32>": "texture_depth_cube",
    "texture_cube_array<f32>": "texture_depth_cube_array",
}
_CMP_CALL_RE = re.compile(r"textureSampleCompareLevel\(\((\w+)\)")


def _fixup_wgsl_depth_textures(wgsl_path: Path) -> bool:
    """Patch Slang WGSL output so depth-comparison textures use the correct type."""
    text = wgsl_path.read_text(encoding="utf-8")

    depth_vars = set(_CMP_CALL_RE.findall(text))
    if not depth_vars:
        return False

    changed = False
    for var in depth_vars:
        for old_type, new_type in _DEPTH_TYPE_MAP.items():
            old = f"{var} : {old_type}"
            new = f"{var} : {new_type}"
            if old in text:
                text = text.replace(old, new)
                changed = True

    if changed:
        wgsl_path.write_text(text, encoding="utf-8")
        logger.info(f"Fixed WGSL depth texture types in {wgsl_path.name}")
    return changed


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

    def format_mcp_output(
        self, records: list[McpLogRecord], returncode: int
    ) -> str | None:
        """Show only summary and errors, skip WGSL output."""
        lines: list[str] = []
        for r in records:
            if r.level in ("error", "critical", "warning"):
                lines.append(r.message)
            elif any(k in r.message for k in ("compiled", "skipped", "emitted")):
                lines.append(r.message)
        if not lines:
            return None
        lines.append("\nFull log: _build/logs/mcp/slangc.log")
        return "\n".join(lines)

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        """Compile Slang shaders configured in config.yaml."""
        root = ctx.workspace_root
        config = ctx.config
        tokens = ctx.tokens

        compiler_path = args.get("compiler_path")
        if compiler_path:
            compiler = str(resolve_path(root, str(compiler_path), tokens))
        else:
            compiler = "slangc"

        conanbuild = Path(tokens["build_dir"]) / "conanbuild"

        search_paths_raw = args.get("search_paths", [])
        search_paths = [resolve_path(root, p, tokens) for p in search_paths_raw]

        shaders, errors = _resolve_slang_shaders(root, config, tokens, args)
        if errors:
            logger.error(f"Shader resolution failed with {errors} error(s)")
            sys.exit(1)
        if not shaders:
            logger.warning("No Slang shaders configured.")
            return

        logs_dir = Path(tokens["logs_root"])
        logs_dir.mkdir(parents=True, exist_ok=True)

        compiled = 0
        skipped = 0
        for input_path, output_path, reflect, defines in shaders:
            if not input_path.exists():
                logger.error(f"Shader input not found: {input_path}")
                sys.exit(1)

            if _should_compile_shader(input_path, output_path, args["force"], search_paths):
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
                for d in defines:
                    cmd.extend(["-D", d])
                for sp in search_paths:
                    cmd.extend(["-I", str(sp)])
                cmd.extend(ctx.passthrough_args)
                shell_cmd = ShellCommand(cmd, env_script=conanbuild)
                try:
                    shell_cmd.exec(log_file=log_file)
                except SystemExit as e:
                    log_content = ""
                    if log_file.exists():
                        log_content = log_file.read_text().strip()
                    if log_content:
                        logger.error(f"slangc failed compiling {input_path} (exit {e.code}):")
                        logger.error(log_content)
                    else:
                        logger.error(
                            f"slangc failed compiling {input_path} "
                            f"(exit {e.code}, no output)"
                        )
                    logger.error(f"Command: {' '.join(cmd)}")
                    raise
                _fixup_wgsl_depth_textures(output_path)
                compiled += 1
            else:
                _fixup_wgsl_depth_textures(output_path)
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
                        conanbuild, ctx.passthrough_args, search_paths,
                    )

        logger.info(f"slangc compiled {compiled} shader(s)")
        if skipped:
            logger.info(f"slangc skipped {skipped} up-to-date shader(s)")
