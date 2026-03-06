"""Shader reflection codegen — generates C++ headers from reflection JSON."""

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


def _visibility_flags(visibility: list[str]) -> str:
    """Convert visibility list to WGPUShaderStage flags expression."""
    parts = []
    if "vertex" in visibility:
        parts.append("WGPUShaderStage_Vertex")
    if "fragment" in visibility:
        parts.append("WGPUShaderStage_Fragment")
    if not parts:
        parts.append("WGPUShaderStage_Vertex | WGPUShaderStage_Fragment")
    return " | ".join(parts)


def _buffer_type_field(buffer_type: str) -> str:
    """Return the struct field assignment for buffer type."""
    mapping = {
        "WGPUBufferBindingType_Uniform": "Uniform",
        "WGPUBufferBindingType_Storage": "Storage",
        "WGPUBufferBindingType_ReadOnlyStorage": "ReadOnlyStorage",
    }
    return mapping.get(buffer_type, "Uniform")


def _build_template_data(reflection: dict, namespace: str) -> dict:
    """Transform reflection JSON into template context variables."""
    ep = reflection.get("entry_points", {})
    vertex_entry = ep.get("vertex", "vs_main")
    fragment_entry = ep.get("fragment", "fs_main")

    # Vertex layout
    vertex_inputs = reflection.get("vertex_inputs", [])
    vertex_layout = None
    if vertex_inputs:
        # Compute stride and offsets
        offset = 0
        attrs = []
        for vi in sorted(vertex_inputs, key=lambda x: x["location"]):
            attrs.append({
                "format": vi.get("format", ""),
                "offset": offset,
                "location": vi["location"],
                "name": vi.get("name", ""),
            })
            offset += vi.get("size", 0)

        vertex_layout = {
            "stride": offset,
            "attributes": attrs,
        }

    # Bind groups — group bindings by group number
    bindings = reflection.get("bindings", [])
    groups: dict[int, list[dict]] = {}
    for b in bindings:
        g = b["group"]
        groups.setdefault(g, []).append(b)

    bind_groups = []
    for group_num in sorted(groups):
        entries = sorted(groups[group_num], key=lambda x: x["binding"])
        bind_groups.append({
            "group": group_num,
            "entries": [
                {
                    "binding": e["binding"],
                    "visibility": _visibility_flags(e.get("visibility", [])),
                    "buffer_type": _buffer_type_field(e.get("buffer_type", "")),
                    "min_binding_size": e.get("struct_size", 0),
                    "var_name": e.get("var_name", ""),
                    "type_name": e.get("type_name", ""),
                }
                for e in entries
            ],
        })

    # Fragment outputs
    fragment_outputs = reflection.get("fragment_outputs", [])
    color_attachment_count = len(fragment_outputs)

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
