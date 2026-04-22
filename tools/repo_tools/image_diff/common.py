"""Shared helpers for ``image-diff`` and ``bake-gt`` tools.

Loads the ``image_diff`` section of ``config.yaml`` into typed dataclasses,
resolves paths, and runs the editor via the ``./repo launch`` entrypoint.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repo_tools.core import is_windows, logger


@dataclass(frozen=True)
class Case:
    """A single diff case declared in ``config.yaml``."""

    name: str
    scene: Path
    camera: str
    renderer: str
    gt: Path
    threshold: float


@dataclass(frozen=True)
class BakeConfig:
    """``image_diff.gt_bake`` section."""

    renderer: str
    frames: int


@dataclass(frozen=True)
class ImageDiffConfig:
    """Fully-resolved ``image_diff`` section."""

    tile_size: int
    default_threshold: float
    gt_dir: Path
    out_dir: Path
    gt_bake: BakeConfig
    cases: list[Case]


def _require(cfg: dict[str, Any], key: str, where: str) -> Any:
    if key not in cfg:
        raise KeyError(f"image_diff: missing required key '{key}' in {where}")
    return cfg[key]


def load_image_diff_config(
    workspace_root: Path, config: dict[str, Any],
) -> ImageDiffConfig:
    """Parse ``config['image_diff']`` into :class:`ImageDiffConfig`.

    All paths in the returned config are absolute and rooted at
    *workspace_root*. Raises ``KeyError`` / ``ValueError`` on missing or
    malformed entries -- fail loud per project conventions.
    """
    section = config.get("image_diff")
    if not isinstance(section, dict):
        raise KeyError(
            "image_diff: missing top-level 'image_diff' section in config.yaml"
        )

    tile_size = int(section.get("tile_size", 64))
    if tile_size <= 0:
        raise ValueError(
            f"image_diff.tile_size must be positive (got {tile_size})"
        )
    default_threshold = float(_require(section, "default_threshold", "image_diff"))
    gt_dir = workspace_root / str(_require(section, "gt_dir", "image_diff"))
    out_dir = workspace_root / str(_require(section, "out_dir", "image_diff"))

    bake_raw = _require(section, "gt_bake", "image_diff")
    if not isinstance(bake_raw, dict):
        raise ValueError("image_diff.gt_bake must be a mapping")
    bake = BakeConfig(
        renderer=str(_require(bake_raw, "renderer", "image_diff.gt_bake")),
        frames=int(_require(bake_raw, "frames", "image_diff.gt_bake")),
    )

    cases_raw = _require(section, "cases", "image_diff")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("image_diff.cases must be a non-empty list")

    cases: list[Case] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(cases_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"image_diff.cases[{idx}] must be a mapping")
        name = str(_require(entry, "name", f"image_diff.cases[{idx}]"))
        if name in seen_names:
            raise ValueError(f"image_diff.cases: duplicate case name '{name}'")
        seen_names.add(name)
        cases.append(Case(
            name=name,
            scene=workspace_root / str(_require(entry, "scene", name)),
            camera=str(_require(entry, "camera", name)),
            renderer=str(_require(entry, "renderer", name)),
            gt=gt_dir / str(_require(entry, "gt", name)),
            threshold=float(entry.get("threshold", default_threshold)),
        ))
    return ImageDiffConfig(
        tile_size=tile_size,
        default_threshold=default_threshold,
        gt_dir=gt_dir,
        out_dir=out_dir,
        gt_bake=bake,
        cases=cases,
    )


def select_cases(cfg: ImageDiffConfig, case_name: str | None) -> list[Case]:
    """Return all cases or just the named one. Raises on unknown name."""
    if case_name is None:
        return list(cfg.cases)
    for c in cfg.cases:
        if c.name == case_name:
            return [c]
    known = ", ".join(c.name for c in cfg.cases)
    raise KeyError(f"image_diff: unknown case '{case_name}' (known: {known})")


def build_editor_args(
    case: Case,
    capture_path: Path,
    renderer: str,
    frames: int,
    from_package: bool = False,
) -> list[str]:
    """Assemble ``launch`` arguments for a capture run.

    ``--from-package`` is a ``launch``-subcommand flag and must precede the
    ``editor`` positional, so it's assembled here alongside the rest of the
    argv rather than bolted on inside :func:`run_launch`.
    """
    args: list[str] = []
    if from_package:
        args.append("--from-package")
    args += [
        "editor",
        f"--capture-and-quit={capture_path}",
        "--frames", str(frames),
        "--usd", str(case.scene),
        "--camera", case.camera,
        "--renderer", renderer,
    ]
    return args


def run_launch(
    workspace_root: Path,
    launch_args: list[str],
    build_type: str,
    log_file: Path | None = None,
) -> None:
    """Shell out to ``./repo launch`` with *launch_args*.

    Uses the top-level ``./repo`` shim so the ``launch`` tool resolves its
    own Conan env and runtime DLLs -- we must not re-implement that here
    (see ticket: "no duplication of launch logic").
    """
    shim = workspace_root / ("repo.cmd" if is_windows() else "repo")
    if not shim.exists():
        raise FileNotFoundError(
            f"./repo shim not found at {shim}; bootstrap the framework first"
        )
    cmd = [str(shim), "--build-type", build_type, "launch", *launch_args]
    # MSYS on Windows rewrites /Root/... paths to backslashed Windows paths
    # when handing CLI args to native exes. Disable that for camera prim
    # paths so the editor sees "/Root/Camera" verbatim.
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    logger.info(f"$ {shim.name} launch {' '.join(launch_args)}")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()
        rc = proc.returncode
    else:
        rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(
            f"launch editor failed (exit {rc}) -- args: {launch_args}"
        )


def require_flip_evaluator() -> Any:
    """Import ``flip_evaluator`` or fail loud.

    We do NOT try to ``pip install`` on the fly -- the dep is declared in
    ``repo.extra_deps`` and should have been pulled in by ``./repo init``.
    """
    try:
        import flip_evaluator  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "flip-evaluator is not installed. The image-diff suite requires it. "
            "Ensure 'flip-evaluator' is listed in config.yaml under "
            "repo.extra_deps and run './repo init' to install it."
        ) from exc
    return flip_evaluator
