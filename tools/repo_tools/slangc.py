"""Slang shader compilation command."""

import argparse
import glob
import sys
from pathlib import Path
import shutil

from repo_tools import (
    RepoContext,
    RepoTool,
    build_repo_context,
    load_repo_config,
    logger,
    print_tool,
    run_command,
    resolve_path,
    is_windows,
)


def _expand_glob_paths(pattern: Path) -> list[Path]:
    pattern_text = str(pattern)
    if any(char in pattern_text for char in ("*", "?", "[")):
        return sorted(Path(match) for match in glob.glob(pattern_text, recursive=True))
    return [pattern]


def _resolve_slang_shaders(
    root: Path, config: dict, context: RepoContext, args: argparse.Namespace
) -> tuple[list[tuple[Path, Path]], int]:
    shaders = getattr(args, "shaders", None)
    if shaders is None:
        shaders = config.get("slangc", {}).get("shaders", [])

    if not shaders:
        return [], 0
    if not isinstance(shaders, list):
        logger.warning("Slang shader configuration must be a list.")
        return [], 0

    resolved: list[tuple[Path, Path]] = []
    errors = 0
    seen_outputs: set[Path] = set()

    for shader in shaders:
        if not isinstance(shader, dict):
            continue
        input_value = shader.get("input")
        if not input_value:
            continue
        output_value = shader.get("output")

        input_pattern = resolve_path(root, str(input_value), context)
        input_paths = [
            path for path in _expand_glob_paths(input_pattern) if path.is_file()
        ]
        if not input_paths:
            logger.error(f"No shader inputs matched: {input_pattern}")
            errors += 1
            continue

        output_pattern_text = None
        if output_value:
            output_pattern_text = str(resolve_path(root, str(output_value), context))
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
            resolved.append((input_path, output_path))

    return resolved, errors


def _find_slangc(
    root: Path, config: dict, context: RepoContext, args: argparse.Namespace
) -> Path | None:
    compiler_path = getattr(args, "compiler_path", None)
    if compiler_path is None:
        compiler_path = config.get("slangc", {}).get("compiler_path")
    if compiler_path:
        path = resolve_path(root, str(compiler_path), context)
        if path.exists():
            return path

    search_roots = getattr(args, "compiler_search_roots", None)
    if search_roots is None:
        search_roots = config.get("slangc", {}).get("compiler_search_roots", [])
    if isinstance(search_roots, (str, Path)):
        search_roots = [search_roots]

    candidates: list[Path] = []
    for entry in search_roots or []:
        candidate = resolve_path(root, str(entry), context)
        for match in _expand_glob_paths(candidate):
            if match.exists():
                candidates.append(match)

    exe_name = "slangc.exe" if is_windows() else "slangc"
    for candidate_root in candidates:
        if candidate_root.is_file():
            if candidate_root.name == exe_name:
                return candidate_root
            continue
        for path in candidate_root.rglob(exe_name):
            return path

    exe_path = shutil.which(exe_name)
    if exe_path:
        return Path(exe_path)
    return None


def _should_compile_shader(input_path: Path, output_path: Path, force: bool) -> bool:
    if force:
        return True
    if not output_path.exists():
        return True
    return output_path.stat().st_mtime < input_path.stat().st_mtime


class SlangcTool(RepoTool):
    name = "slangc"
    help = "Compile Slang shaders"

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
            help="Recompile shaders even if outputs are up to date",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(
            build_type=context["build_type"], force=False, passthrough_args=[]
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Compile Slang shaders configured in config.yaml."""
        root = Path(__file__).parent.parent.parent
        config = load_repo_config(root)
        context = build_repo_context(root, args.build_type, config)
        compiler = _find_slangc(root, config, context, args)

        if compiler is None:
            logger.error("slangc compiler not found. Run '.\\pts.cmd build' first.")
            sys.exit(1)

        shaders, errors = _resolve_slang_shaders(root, config, context, args)
        if errors:
            sys.exit(1)
        if not shaders:
            logger.warning("No Slang shaders configured.")
            return

        logs_dir = Path(context["logs_root"])
        logs_dir.mkdir(parents=True, exist_ok=True)

        compiled = 0
        skipped = 0
        for input_path, output_path in shaders:
            if not input_path.exists():
                logger.error(f"Shader input not found: {input_path}")
                sys.exit(1)

            if not _should_compile_shader(input_path, output_path, args.force):
                logger.info(f"Skipping up-to-date shader: {input_path}")
                skipped += 1
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"slangc_{output_path.stem}.log"
            cmd = [
                str(compiler),
                str(input_path),
                "-o",
                str(output_path),
                "-target",
                "wgsl",
            ]
            cmd.extend(getattr(args, "passthrough_args", []))
            run_command(cmd, log_file=log_file)
            compiled += 1

        print_tool(f"slangc compiled {compiled} shader(s)")
        if skipped:
            print_tool(f"slangc skipped {skipped} up-to-date shader(s)")


