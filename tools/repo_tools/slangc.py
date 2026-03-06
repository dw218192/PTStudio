"""Slang shader compilation command."""

import json
import re
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


# ── WGSL type metadata ──────────────────────────────────────────────

_WGSL_TYPE_SIZE: dict[str, int] = {
    "f32": 4, "i32": 4, "u32": 4,
    "vec2<f32>": 8, "vec2<i32>": 8, "vec2<u32>": 8,
    "vec3<f32>": 12, "vec3<i32>": 12, "vec3<u32>": 12,
    "vec4<f32>": 16, "vec4<i32>": 16, "vec4<u32>": 16,
    "mat2x2<f32>": 16, "mat3x3<f32>": 48, "mat4x4<f32>": 64,
}

_WGSL_TYPE_ALIGN: dict[str, int] = {
    "f32": 4, "i32": 4, "u32": 4,
    "vec2<f32>": 8, "vec2<i32>": 8, "vec2<u32>": 8,
    "vec3<f32>": 16, "vec3<i32>": 16, "vec3<u32>": 16,
    "vec4<f32>": 16, "vec4<i32>": 16, "vec4<u32>": 16,
    "mat2x2<f32>": 8, "mat3x3<f32>": 16, "mat4x4<f32>": 16,
}

_WGSL_VERTEX_FORMAT: dict[str, str] = {
    "f32": "WGPUVertexFormat_Float32",
    "vec2<f32>": "WGPUVertexFormat_Float32x2",
    "vec3<f32>": "WGPUVertexFormat_Float32x3",
    "vec4<f32>": "WGPUVertexFormat_Float32x4",
    "i32": "WGPUVertexFormat_Sint32",
    "vec2<i32>": "WGPUVertexFormat_Sint32x2",
    "vec3<i32>": "WGPUVertexFormat_Sint32x3",
    "vec4<i32>": "WGPUVertexFormat_Sint32x4",
    "u32": "WGPUVertexFormat_Uint32",
    "vec2<u32>": "WGPUVertexFormat_Uint32x2",
    "vec3<u32>": "WGPUVertexFormat_Uint32x3",
    "vec4<u32>": "WGPUVertexFormat_Uint32x4",
}


# ── WGSL reflection parser ──────────────────────────────────────────


def _parse_wgsl_structs(wgsl: str) -> dict[str, list[dict]]:
    """Extract struct definitions from WGSL, returning {name: [fields]}."""
    structs: dict[str, list[dict]] = {}
    for m in re.finditer(r'struct\s+(\w+)\s*\{([^}]+)\}', wgsl):
        name = m.group(1)
        body = m.group(2)
        fields: list[dict] = []
        for fm in re.finditer(
            r'(?:@(\w+)\(([^)]+)\)\s+)?(\w+)\s*:\s*([^,\n]+)', body
        ):
            attr = fm.group(1)  # "location" or "builtin" or None
            attr_val = fm.group(2)
            field_name = fm.group(3)
            field_type = fm.group(4).strip().rstrip(',')
            field: dict = {"name": field_name, "type": field_type}
            if attr == "location" and attr_val is not None:
                field["location"] = int(attr_val)
            elif attr == "builtin":
                field["builtin"] = attr_val
            fields.append(field)
        structs[name] = fields
    return structs


def _compute_struct_size(fields: list[dict]) -> int:
    """Compute byte size of a WGSL struct (uniform buffer layout rules)."""
    offset = 0
    max_align = 0
    for f in fields:
        t = f["type"]
        size = _WGSL_TYPE_SIZE.get(t)
        align = _WGSL_TYPE_ALIGN.get(t)
        if size is None or align is None:
            continue
        offset = (offset + align - 1) & ~(align - 1)
        offset += size
        max_align = max(max_align, align)
    # Uniform struct alignment is at least 16
    struct_align = max(max_align, 16)
    return (offset + struct_align - 1) & ~(struct_align - 1)


def _extract_function_body(wgsl: str, fn_name: str) -> str:
    """Extract the body text of a function by name (simple brace matching)."""
    pattern = rf'fn\s+{re.escape(fn_name)}\s*\([^)]*\)[^{{]*\{{'
    m = re.search(pattern, wgsl)
    if not m:
        return ""
    start = m.end()
    depth = 1
    pos = start
    while pos < len(wgsl) and depth > 0:
        if wgsl[pos] == '{':
            depth += 1
        elif wgsl[pos] == '}':
            depth -= 1
        pos += 1
    return wgsl[start:pos - 1]


def parse_wgsl_reflection(wgsl: str) -> dict:
    """Parse WGSL text and extract reflection data as a JSON-serializable dict."""
    structs = _parse_wgsl_structs(wgsl)

    # ── Entry points ──
    vertex_entry = None
    vertex_input_struct = None
    vm = re.search(r'@vertex\s+fn\s+(\w+)\s*\(\s*(\w+)\s*:\s*(\w+)\s*\)', wgsl)
    if vm:
        vertex_entry = vm.group(1)
        vertex_input_struct = vm.group(3)

    fragment_entry = None
    fragment_outputs: list[dict] = []
    fm = re.search(r'@fragment\s+fn\s+(\w+)\s*\(([^)]*)\)\s*->\s*(.+?)\s*\{', wgsl)
    if fm:
        fragment_entry = fm.group(1)
        return_type = fm.group(3).strip()
        # Check for @location annotated return
        loc_m = re.match(r'@location\((\d+)\)\s+(.+)', return_type)
        if loc_m:
            fragment_outputs.append({
                "location": int(loc_m.group(1)),
                "type": loc_m.group(2).strip(),
            })
        elif return_type in structs:
            for f in structs[return_type]:
                if "location" in f:
                    fragment_outputs.append({
                        "location": f["location"],
                        "type": f["type"],
                    })

    # ── Vertex inputs ──
    vertex_inputs: list[dict] = []
    if vertex_input_struct and vertex_input_struct in structs:
        for f in structs[vertex_input_struct]:
            if "location" in f:
                vertex_inputs.append({
                    "location": f["location"],
                    "name": f["name"],
                    "type": f["type"],
                    "format": _WGSL_VERTEX_FORMAT.get(f["type"], ""),
                    "size": _WGSL_TYPE_SIZE.get(f["type"], 0),
                })

    # ── Bindings ──
    bindings: list[dict] = []
    vertex_body = _extract_function_body(wgsl, vertex_entry) if vertex_entry else ""
    fragment_body = _extract_function_body(wgsl, fragment_entry) if fragment_entry else ""

    for bm in re.finditer(
        r'@group\((\d+)\)\s+@binding\((\d+)\)\s+var<(\w+)>\s+(\w+)\s*:\s*(\w+)',
        wgsl,
    ):
        group = int(bm.group(1))
        binding = int(bm.group(2))
        buf_type = bm.group(3)  # "uniform", "storage", etc.
        var_name = bm.group(4)
        type_name = bm.group(5)

        # Compute struct size
        struct_size = 0
        if type_name in structs:
            struct_size = _compute_struct_size(structs[type_name])

        # Determine visibility by checking variable usage in entry point bodies
        visibility: list[str] = []
        if var_name in vertex_body:
            visibility.append("vertex")
        if var_name in fragment_body:
            visibility.append("fragment")
        if not visibility:
            visibility = ["vertex", "fragment"]

        buffer_type_map = {
            "uniform": "WGPUBufferBindingType_Uniform",
            "storage": "WGPUBufferBindingType_Storage",
        }

        bindings.append({
            "group": group,
            "binding": binding,
            "buffer_type": buffer_type_map.get(buf_type, "WGPUBufferBindingType_Uniform"),
            "var_name": var_name,
            "type_name": type_name,
            "struct_size": struct_size,
            "visibility": visibility,
        })

    return {
        "entry_points": {
            "vertex": vertex_entry,
            "fragment": fragment_entry,
        },
        "vertex_inputs": vertex_inputs,
        "bindings": bindings,
        "fragment_outputs": fragment_outputs,
    }


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
    """Emit reflection JSON for a compiled shader.

    Tries slangc -emit-reflection-json first. On failure, parses the WGSL
    output to produce equivalent reflection data.
    """
    reflect_path = output_path.with_suffix(".reflect.json")

    # Try native slangc reflection
    reflect_cmd = [
        compiler,
        str(input_path),
        "-target", "wgsl",
        "-emit-reflection-json", str(reflect_path),
    ]
    reflect_cmd.extend(passthrough_args)

    result = ShellCommand(reflect_cmd, env_script=conanbuild).run(
        capture_output=True, text=True,
    )

    if result.returncode == 0 and reflect_path.exists():
        logger.info(f"slangc emitted reflection JSON: {reflect_path}")
        return

    # Fallback: parse WGSL output
    logger.info(
        f"slangc -emit-reflection-json not available, "
        f"parsing WGSL for reflection: {output_path}"
    )
    wgsl_text = output_path.read_text(encoding="utf-8")
    reflection = parse_wgsl_reflection(wgsl_text)

    reflect_path.parent.mkdir(parents=True, exist_ok=True)
    reflect_path.write_text(
        json.dumps(reflection, indent=2), encoding="utf-8"
    )
    logger.info(f"Generated reflection JSON from WGSL: {reflect_path}")


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
