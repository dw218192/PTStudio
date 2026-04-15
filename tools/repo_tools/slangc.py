"""Slang shader compilation driver.

Thin driver over the `pts_shaderc` CLI (tools/pts_shaderc/). Resolves
config.yaml `slangc.shaders` entries -- glob expansion, variant suffixes,
optional metadata-header emission -- and invokes pts_shaderc once per
(input x variant). pts_shaderc handles compile, metadata-header emission,
and staleness checks in-process via libslang.
"""

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


def _resolve_pts_shaderc(build_dir: Path) -> Path:
    """Locate the staged pts_shaderc binary.

    pts_shaderc is a native host tool. In a same-platform build it sits in
    the current build_dir's bin/. In a cross-compile (e.g. Emscripten
    target) the host binary lives under a sibling platform's build dir --
    it was staged there by a preceding `./repo build --host-tools-only`
    invocation for the host platform. We search both.
    """
    exe = "pts_shaderc.exe" if sys.platform == "win32" else "pts_shaderc"

    direct = build_dir / "bin" / exe
    if direct.exists():
        return direct

    # Fall back to any sibling platform build dir. `_build/<platform>/<cfg>/`
    # -- two levels up from build_dir is `_build/`, sibling platform dirs
    # are at the same depth.
    build_root = build_dir.parent.parent
    if build_root.is_dir():
        for candidate in build_root.glob(f"*/{build_dir.name}/bin/{exe}"):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

    raise FileNotFoundError(
        f"pts_shaderc not found under {build_dir} or any sibling platform "
        f"build dir; run `./repo build --host-tools-only` first"
    )


def _insert_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(path.stem + suffix + path.suffix)


def _variants(shader: dict) -> list[dict]:
    cfg = shader.get("variants")
    if cfg is None:
        return [{"defines": list(shader.get("defines", [])), "suffix": ""}]
    out: list[dict] = []
    for v in cfg:
        if not isinstance(v, dict):
            raise ValueError(f"variant must be a dict, got {type(v).__name__}")
        out.append({"defines": list(v.get("defines", [])), "suffix": str(v.get("suffix", ""))})
    return out


class SlangcTool(RepoTool):
    name = "slangc"
    help = "Compile Slang shaders via pts_shaderc (WGSL + optional metadata header)"

    def setup(self, cmd: click.Command) -> click.Command:
        return click.option(
            "-f", "--force", is_flag=True, default=None,
            help="Recompile even if outputs are up to date",
        )(cmd)

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {"force": False}

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        root = ctx.workspace_root
        config = ctx.config
        tokens = ctx.tokens
        force = bool(args.get("force"))

        slangc_cfg = config.get("slangc", {}) or {}
        shaders = slangc_cfg.get("shaders") or []
        if not shaders:
            logger.warning("No Slang shaders configured.")
            return

        build_dir = Path(tokens["build_dir"])
        pts_shaderc = _resolve_pts_shaderc(build_dir)
        # Resolve the conanrun env script for pts_shaderc. Windows uses
        # PATH (not RPATH) for DLL discovery, so sourcing conanrun is
        # required there to find slang.dll. Linux/macOS get away without
        # it because Conan bakes RPATH at build time, but we still use
        # conanrun when present.
        #
        # Two candidates, in preference order:
        #   1. Next to the binary: `<pts_shaderc_dir>/conanrun.*`
        #      (staged during `--host-tools-only` so cross-compile works)
        #   2. The current build's main conanrun: `<build_dir>/conanrun.*`
        #      (present in native same-platform builds)
        conanrun_suffix = ".bat" if sys.platform == "win32" else ".sh"
        conanrun: Path | None = None
        for candidate in (pts_shaderc.parent, build_dir):
            script = candidate / f"conanrun{conanrun_suffix}"
            if script.exists():
                conanrun = candidate / "conanrun"  # ShellCommand appends suffix
                break
        logs_dir = Path(tokens["logs_root"])
        logs_dir.mkdir(parents=True, exist_ok=True)

        search_paths = [resolve_path(root, p, tokens) for p in slangc_cfg.get("search_paths", [])]
        logger.info(f"slangc: using pts_shaderc ({pts_shaderc})")

        count = 0
        seen_outputs: set[Path] = set()
        for idx, shader in enumerate(shaders):
            if not isinstance(shader, dict):
                raise ValueError(f"Shader entry {idx}: expected dict")
            input_value = shader.get("input")
            output_value = shader.get("output")
            if not input_value or not output_value:
                continue
            metadata = shader.get("metadata")
            variants = _variants(shader)

            input_pattern = resolve_path(root, str(input_value), tokens)
            inputs = sorted(p for p in glob_paths(input_pattern) if p.is_file())
            if not inputs:
                raise FileNotFoundError(f"No shader inputs matched: {input_pattern}")

            output_pattern = str(resolve_path(root, str(output_value), tokens))
            if "*" not in output_pattern and len(inputs) > 1:
                raise ValueError(
                    f"Output path must include '*' when multiple inputs match: {output_pattern}"
                )

            for input_path in inputs:
                base_output = Path(output_pattern.replace("*", input_path.stem))
                for variant in variants:
                    output_path = _insert_suffix(base_output, variant["suffix"])
                    if output_path in seen_outputs:
                        raise ValueError(f"Duplicate shader output path: {output_path}")
                    seen_outputs.add(output_path)

                    cmd = [
                        str(pts_shaderc),
                        "compile",
                        "--source", str(input_path),
                        "--output", str(output_path),
                    ]
                    for d in variant["defines"]:
                        cmd += ["-D", d]
                    for sp in search_paths:
                        cmd += ["-I", str(sp)]
                    # Metadata emits only for the base (no-suffix) variant --
                    # the C++ header is define-agnostic.
                    if metadata and not variant["suffix"]:
                        metadata_output = resolve_path(root, str(metadata["output"]), tokens)
                        cmd += ["--metadata", str(metadata_output),
                                "--namespace", str(metadata["namespace"])]
                    if force:
                        cmd += ["--force"]

                    log_file = logs_dir / f"slangc_{output_path.stem}.log"
                    ShellCommand(cmd, env_script=conanrun).exec(log_file=log_file)
                    count += 1

        logger.info(f"slangc compiled/checked {count} shader variant(s) via pts_shaderc")
