"""Resource embedding tool using Jinja2 templates."""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from jinja2 import Template

from repo_tools import (
    RepoContext,
    RepoTool,
    build_repo_context,
    load_repo_config,
    logger,
    print_tool,
    resolve_path,
)


@cache
def _load_template(template_path: Path) -> Template:
    """Load and cache the Jinja2 template from file."""
    return Template(template_path.read_text(encoding="utf-8"))


@dataclass
class ResourceData:
    """Processed resource data for template rendering."""

    path: str
    identifier: str
    size: int
    is_text: bool
    content: str = ""
    delimiter: str = ""
    hex_data: str = ""


def _expand_glob_paths(pattern: Path) -> list[Path]:
    """Expand glob pattern to list of matching files."""
    pattern_text = str(pattern)
    if any(char in pattern_text for char in ("*", "?", "[")):
        return sorted(Path(match) for match in glob.glob(pattern_text, recursive=True))
    return [pattern]


def _find_common_ancestor(paths: list[Path]) -> Path:
    """Find the common ancestor directory of multiple paths."""
    common = os.path.commonpath([str(p) for p in paths])
    common_path = Path(common)
    if common_path.is_file():
        common_path = common_path.parent
    return common_path


def _sanitize_identifier(name: str) -> str:
    """Convert a filename to a valid C++ identifier."""
    result = re.sub(r"[^a-zA-Z0-9]", "_", name)
    if result and result[0].isdigit():
        result = "_" + result
    return result


def _find_raw_string_delimiter(text: str) -> str:
    """Find a delimiter that doesn't appear in the text for raw string literals."""
    for d in ["", "=", "==", "===", "DELIM", "END", "RAW"]:
        if f'){d}"' not in text and f'R"{d}(' not in text:
            return d
    raise ValueError("Could not find suitable raw string delimiter")


def _is_text_content(data: bytes) -> bool:
    """Check if data is valid UTF-8 text without embedded nulls."""
    try:
        text = data.decode("utf-8")
        return "\x00" not in text
    except UnicodeDecodeError:
        return False


def _process_resource(input_file: Path, base_path: Path) -> ResourceData:
    """Process a single resource file into ResourceData."""
    rel_path = os.path.relpath(input_file, base_path)
    path_str = rel_path.replace("\\", "/")
    identifier = _sanitize_identifier(input_file.stem + "_" + input_file.suffix[1:])
    data = input_file.read_bytes()

    if _is_text_content(data):
        text = data.decode("utf-8")
        # Use length of text (not bytes) since CRLF becomes LF in raw string literals
        return ResourceData(
            path=path_str,
            identifier=identifier,
            size=len(text),
            is_text=True,
            content=text,
            delimiter=_find_raw_string_delimiter(text),
        )
    else:
        hex_values = ", ".join(f"0x{b:02x}" for b in data)
        return ResourceData(
            path=path_str,
            identifier=identifier,
            size=len(data),
            is_text=False,
            hex_data=hex_values,
        )


def _compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(manifest_path: Path) -> dict:
    """Load manifest file. Returns empty dict if file doesn't exist."""
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Save manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _needs_regeneration(
    input_files: list[Path],
    output_path: Path,
    manifest_path: Path,
    force: bool,
) -> tuple[bool, dict]:
    """Check if output needs to be regenerated based on file hashes."""
    if force or not output_path.exists():
        return True, {}

    manifest = _load_manifest(manifest_path)
    old_hashes = manifest.get("hashes", {})
    current_hashes = {str(f): _compute_file_hash(f) for f in input_files}

    if old_hashes != current_hashes:
        return True, current_hashes

    return False, current_hashes


def _generate_embedded_header(
    input_files: list[Path],
    output_path: Path,
    namespace: str,
    base_path: Path,
    template_path: Path,
) -> None:
    """Generate embedded resource header using Jinja2 template."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resources = [_process_resource(f, base_path) for f in input_files]
    template = _load_template(template_path)
    header_content = template.render(namespace=namespace, resources=resources)

    output_path.write_text(header_content, encoding="utf-8")


def _resolve_resource_groups(
    root: Path, config: dict, context: RepoContext, args: argparse.Namespace
) -> list[dict]:
    """Resolve resource groups from config."""
    embed_config = config.get("embed", {})
    resources = getattr(args, "resources", None)
    if resources is None:
        resources = embed_config.get("resources", [])

    if not resources:
        return []

    resolved: list[dict] = []

    for resource in resources:
        input_value = resource["input"]
        output_value = resource["output"]
        namespace_value = resource.get("namespace", "embedded_resources")

        # Handle input as string or array
        if isinstance(input_value, str):
            input_patterns = [input_value]
        else:
            input_patterns = input_value

        # Expand all input patterns
        input_files: list[Path] = []
        for pattern in input_patterns:
            input_pattern = resolve_path(root, str(pattern), context)
            matched = [p for p in _expand_glob_paths(input_pattern) if p.is_file()]
            input_files.extend(matched)

        if not input_files:
            raise ValueError(f"No files matched input patterns: {input_patterns}")

        # Remove duplicates while preserving order
        seen: set[Path] = set()
        unique_files: list[Path] = []
        for f in input_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        input_files = unique_files

        output_path = resolve_path(root, str(output_value), context)
        base_path = _find_common_ancestor(input_files)

        resolved.append(
            {
                "input_files": input_files,
                "output": output_path,
                "namespace": namespace_value,
                "base_path": base_path,
            }
        )

    return resolved


class EmbedTool(RepoTool):
    name = "embed"
    help = "Embed resources as C++ headers"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--build-type",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration type (default: Debug)",
        )
        parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Regenerate all resources even if up to date",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(
            build_type=context["build_type"],
            force=False,
            passthrough_args=[],
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Embed resources as C++ headers."""
        root = Path(__file__).parent.parent.parent
        config = load_repo_config(root)
        context = build_repo_context(root, args.build_type, config)

        resource_groups = _resolve_resource_groups(root, config, context, args)
        if not resource_groups:
            logger.info("No resources configured for embedding.")
            return

        # Get template path from args or config (required)
        embed_config = config.get("embed", {})
        template_path_str = getattr(args, "template", None)
        if template_path_str is None:
            template_path_str = embed_config.get("template")
        if not template_path_str:
            raise ValueError(
                "Template path not specified. Set 'template' in embed config."
            )
        template_path = root / template_path_str
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Centralize manifests in build directory
        manifest_dir = Path(context["build_dir"]) / "embed"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        embedded = 0
        skipped = 0

        for group in resource_groups:
            input_files = group["input_files"]
            output_path = group["output"]
            namespace = group["namespace"]
            base_path = group["base_path"]

            # Use output path relative to root for unique manifest name
            rel_output = output_path.relative_to(root)
            manifest_name = (
                str(rel_output).replace("/", "_").replace("\\", "_") + ".manifest.json"
            )
            manifest_path = manifest_dir / manifest_name
            needs_regen, current_hashes = _needs_regeneration(
                input_files, output_path, manifest_path, args.force
            )

            if not needs_regen:
                logger.info(f"Skipping up-to-date: {output_path}")
                skipped += 1
                continue

            logger.info(f"Embedding {len(input_files)} resource(s) -> {output_path}")

            _generate_embedded_header(
                input_files,
                output_path,
                namespace,
                base_path,
                template_path,
            )

            if not current_hashes:
                current_hashes = {str(f): _compute_file_hash(f) for f in input_files}
            _save_manifest(manifest_path, {"hashes": current_hashes})
            embedded += 1

        print_tool(f"embed generated {embedded} header(s)")
        if skipped:
            print_tool(f"embed skipped {skipped} up-to-date header(s)")
