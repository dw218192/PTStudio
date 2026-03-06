"""Shader reflection codegen — generates C++ headers from Slang reflection JSON."""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import click
from jinja2 import Environment

from repo_tools.core import (
    RepoTool,
    ToolContext,
    logger,
    resolve_path,
)


_jinja_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)


@cache
def _load_template(template_path: Path):
    """Load and cache the Jinja2 template from file."""
    return _jinja_env.from_string(template_path.read_text(encoding="utf-8"))


# ── Slang reflection JSON → template context ────────────────────────


def _slang_type_to_vertex_format(type_info: dict) -> tuple[str, int]:
    """Map a Slang reflection type to (WGPUVertexFormat, byte_size)."""
    kind = type_info.get("kind")

    if kind == "scalar":
        scalar = type_info.get("scalarType")
        scalar_map = {
            "float32": ("WGPUVertexFormat_Float32", 4),
            "int32": ("WGPUVertexFormat_Sint32", 4),
            "uint32": ("WGPUVertexFormat_Uint32", 4),
        }
        if scalar not in scalar_map:
            raise ValueError(f"Unsupported scalar type '{scalar}' in vertex input: {type_info}")
        return scalar_map[scalar]

    if kind == "vector":
        count = type_info["elementCount"]
        scalar = type_info["elementType"]["scalarType"]
        table = {
            ("float32", 2): ("WGPUVertexFormat_Float32x2", 8),
            ("float32", 3): ("WGPUVertexFormat_Float32x3", 12),
            ("float32", 4): ("WGPUVertexFormat_Float32x4", 16),
            ("int32", 2): ("WGPUVertexFormat_Sint32x2", 8),
            ("int32", 3): ("WGPUVertexFormat_Sint32x3", 12),
            ("int32", 4): ("WGPUVertexFormat_Sint32x4", 16),
            ("uint32", 2): ("WGPUVertexFormat_Uint32x2", 8),
            ("uint32", 3): ("WGPUVertexFormat_Uint32x3", 12),
            ("uint32", 4): ("WGPUVertexFormat_Uint32x4", 16),
        }
        key = (scalar, count)
        if key not in table:
            raise ValueError(
                f"Unsupported vector type '{scalar}x{count}' in vertex input: {type_info}"
            )
        return table[key]

    raise ValueError(f"Unsupported vertex input type: {type_info}")


def _extract_vertex_inputs(param: dict) -> list[dict]:
    """Extract vertex input attributes from a Slang entry point parameter."""
    binding = param.get("binding", {})
    if binding.get("kind") != "varyingInput":
        return []

    type_info = param.get("type", {})

    # Struct parameter — each field is a separate vertex attribute
    if type_info.get("kind") == "struct":
        inputs = []
        for field in type_info.get("fields", []):
            fb = field.get("binding", {})
            if fb.get("kind") != "varyingInput":
                continue
            fmt, size = _slang_type_to_vertex_format(field["type"])
            inputs.append({
                "location": fb["index"],
                "name": field["name"],
                "format": fmt,
                "size": size,
            })
        return inputs

    # Scalar/vector parameter — single attribute
    fmt, size = _slang_type_to_vertex_format(type_info)
    return [{
        "location": binding["index"],
        "name": param.get("name", ""),
        "format": fmt,
        "size": size,
    }]


def _binding_struct_size(type_info: dict) -> int:
    """Extract the buffer size from a Slang binding type."""
    # constantBuffer → elementVarLayout.binding.size
    evl = type_info.get("elementVarLayout", {})
    evl_binding = evl.get("binding", {})
    if "size" in evl_binding:
        return evl_binding["size"]
    return 0


def _binding_buffer_type(type_info: dict) -> str:
    """Map Slang type kind to WGPUBufferBindingType suffix."""
    kind = type_info.get("kind", "")
    if kind == "constantBuffer":
        return "Uniform"
    if kind in ("structuredBuffer", "rwStructuredBuffer"):
        return "Storage" if "rw" in kind.lower() else "ReadOnlyStorage"
    return "Uniform"


def _visibility_flags(stages: list[str]) -> str:
    """Convert stage list to WGPUShaderStage flags expression."""
    parts = []
    if "vertex" in stages:
        parts.append("WGPUShaderStage_Vertex")
    if "fragment" in stages:
        parts.append("WGPUShaderStage_Fragment")
    if not parts:
        parts.append("WGPUShaderStage_Vertex | WGPUShaderStage_Fragment")
    return " | ".join(parts)


def _count_fragment_outputs(fragment_ep: dict) -> int:
    """Count color attachment outputs from a fragment entry point."""
    result = fragment_ep.get("result", {})
    result_type = result.get("type", {})

    # Struct return — count fields with varyingOutput binding
    if result_type.get("kind") == "struct":
        count = 0
        for field in result_type.get("fields", []):
            fb = field.get("binding", {})
            if fb.get("kind") == "varyingOutput":
                count += 1
        return max(count, 1)

    # Single output
    rb = result.get("binding", {})
    if rb.get("kind") == "varyingOutput":
        return 1
    return 1


def _build_template_data(reflection: dict, namespace: str) -> dict:
    """Transform Slang reflection JSON into template context variables."""
    entry_points = reflection.get("entryPoints", [])
    vertex_ep = next((ep for ep in entry_points if ep["stage"] == "vertex"), None)
    fragment_ep = next((ep for ep in entry_points if ep["stage"] == "fragment"), None)

    vertex_entry = vertex_ep["name"] if vertex_ep else "vs_main"
    fragment_entry = fragment_ep["name"] if fragment_ep else "fs_main"

    # ── Vertex layout ──
    vertex_layout = None
    if vertex_ep:
        all_inputs = []
        for param in vertex_ep.get("parameters", []):
            all_inputs.extend(_extract_vertex_inputs(param))

        if all_inputs:
            all_inputs.sort(key=lambda x: x["location"])
            offset = 0
            attrs = []
            for vi in all_inputs:
                attrs.append({
                    "format": vi["format"],
                    "offset": offset,
                    "location": vi["location"],
                    "name": vi["name"],
                })
                offset += vi["size"]
            vertex_layout = {"stride": offset, "attributes": attrs}

    # ── Bind groups ──
    # Top-level parameters are global bindings (uniforms, storage buffers, etc.)
    global_params = reflection.get("parameters", [])
    # Group 0 is the default; Slang uses registerSpace for multi-group layouts
    groups: dict[int, list[dict]] = {}

    for param in global_params:
        pb = param.get("binding", {})
        if pb.get("kind") != "descriptorTableSlot":
            continue

        binding_idx = pb.get("index", 0)
        # registerSpace for group, defaulting to 0
        group_idx = pb.get("space", 0)
        type_info = param.get("type", {})

        # Determine visibility from entry point usage
        visibility = []
        for ep in entry_points:
            for b in ep.get("bindings", []):
                if b["name"] == param["name"] and b["binding"].get("used", 0):
                    visibility.append(ep["stage"])

        entry = {
            "binding": binding_idx,
            "visibility": _visibility_flags(visibility),
            "buffer_type": _binding_buffer_type(type_info),
            "min_binding_size": _binding_struct_size(type_info),
            "var_name": param.get("name", ""),
            "type_name": type_info.get("elementType", {}).get("name", ""),
        }
        groups.setdefault(group_idx, []).append(entry)

    bind_groups = []
    for group_num in sorted(groups):
        entries = sorted(groups[group_num], key=lambda x: x["binding"])
        bind_groups.append({"group": group_num, "entries": entries})

    # ── Fragment outputs ──
    color_attachment_count = _count_fragment_outputs(fragment_ep) if fragment_ep else 1

    return {
        "namespace": namespace,
        "vertex_entry": vertex_entry,
        "fragment_entry": fragment_entry,
        "vertex_layout": vertex_layout,
        "bind_groups": bind_groups,
        "color_attachment_count": color_attachment_count,
    }


class ShaderCodegenTool(RepoTool):
    name = "shader_codegen"
    help = "Generate C++ headers from shader reflection data"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "-f",
            "--force",
            is_flag=True,
            default=None,
            help="Regenerate headers even if inputs are up to date",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "force": False,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        """Generate C++ headers from shader reflection JSON."""
        root = ctx.workspace_root
        tokens = ctx.tokens
        force = args.get("force", False)

        shaders = args.get("shaders")
        if not shaders:
            logger.warning("No shader_codegen shaders configured.")
            return

        # Resolve template path
        template_path_str = args.get("template")
        if not template_path_str:
            template_path_str = "core/templates/shader_metadata.h.j2"
        template_path = root / template_path_str
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        generated = 0
        skipped = 0

        for shader in shaders:
            reflect_value = shader.get("reflect")
            if not reflect_value:
                continue

            reflect_path = resolve_path(root, str(reflect_value), tokens)
            output_path = resolve_path(root, str(shader["output"]), tokens)
            namespace = shader.get("namespace", "shader_metadata")

            if not reflect_path.exists():
                logger.error(f"Reflection JSON not found: {reflect_path}")
                sys.exit(1)

            # Skip if output is up to date
            if (
                not force
                and output_path.exists()
                and output_path.stat().st_mtime >= reflect_path.stat().st_mtime
            ):
                logger.info(f"Skipping up-to-date: {output_path}")
                skipped += 1
                continue

            # Load reflection data
            reflection = json.loads(reflect_path.read_text(encoding="utf-8"))

            # Build template context and render
            tmpl_data = _build_template_data(reflection, namespace)
            template = _load_template(template_path)
            header_content = template.render(**tmpl_data)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(header_content, encoding="utf-8")
            logger.info(f"Generated shader metadata: {output_path}")
            generated += 1

        logger.info(f"shader_codegen generated {generated} header(s)")
        if skipped:
            logger.info(f"shader_codegen skipped {skipped} up-to-date header(s)")
