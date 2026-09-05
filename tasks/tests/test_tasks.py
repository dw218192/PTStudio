"""Exercise the direct task entry points and their shared project behavior."""

import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from tasks import clean, package
from tasks.image_diff_support import run_launch
from tasks.utils.project import ROOT, load_context

MANIFEST = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
COMMANDS = [
    (name, command)
    for name, command in MANIFEST["tasks"].items()
    if isinstance(command, str) and command.startswith("python -m tasks.")
]


@pytest.mark.parametrize(("name", "command"), COMMANDS, ids=[name for name, _ in COMMANDS])
def test_direct_task_help(name, command):
    argv = shlex.split(command)
    result = subprocess.run(
        [sys.executable, *argv[1:], "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, f"{name}: {result.stdout}\n{result.stderr}"
    assert "--help" in result.stdout


def write_config(root: Path, **sections):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "paths": {
            "build_root": "{workspace_root}/output",
            "build_dir": "{build_root}/{platform}/{build_type}",
            "logs_root": "{workspace_root}/logs",
        },
        **sections,
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_context_expands_paths_filters_and_local_overrides(tmp_path):
    write_config(
        tmp_path,
        embed={"input": ["native.wgsl"], "input@emscripten": ["web.wgsl"]},
        build={"windowing": "glfw", "usd_modules": ["usdGeom"]},
    )
    (tmp_path / "config.local.yaml").write_text(
        "build:\n  windowing: 'null'\n  usd_modules+: [usdShade]\n", encoding="utf-8"
    )
    native = load_context("windows-x64", "release", root=tmp_path)
    web = load_context("emscripten", "Release", root=tmp_path)
    assert Path(native.tokens["build_dir"]) == tmp_path / "output/windows-x64/Release"
    assert Path(web.tokens["build_dir"]) == tmp_path / "output/emscripten/Release"
    assert native.section("embed")["input"] == ["native.wgsl"]
    assert web.section("embed")["input"] == ["web.wgsl"]
    assert native.section("build")["usd_modules"] == ["usdGeom", "usdShade"]
    assert native.arguments("build", {}, {"windowing": None})["windowing"] == "null"
    assert native.arguments("build", {}, {"windowing": "glfw"})["windowing"] == "glfw"


def test_image_capture_launcher_runs_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "launch.log"
    # Exercise the real child module, interpreter, cwd and output forwarding.
    run_launch(ROOT, ["--help"], "Debug", log)
    assert "Launch executables" in log.read_text(encoding="utf-8")


def test_package_preserves_nested_layout_and_rejects_missing_inputs(tmp_path):
    write_config(
        tmp_path,
        package={
            "output_dir": "{workspace_root}/package",
            "mappings": [{"src": "{build_dir}/**/*", "dest": "bin"}],
        },
    )
    context = load_context("linux-x64", root=tmp_path)
    source = Path(context.tokens["build_dir"]) / "nested/example.txt"
    source.parent.mkdir(parents=True)
    source.write_text("packaged content", encoding="utf-8")
    package.run(context, {})
    target = tmp_path / "package/bin/nested/example.txt"
    assert target.read_text(encoding="utf-8") == "packaged content"
    source.unlink()
    with pytest.raises(SystemExit) as error:
        package.run(context, {"no_clean": True})
    assert error.value.code == 1
    assert target.exists()


def test_clean_rejects_outside_workspace_root_and_protected_paths(tmp_path):
    workspace = tmp_path / "workspace"
    write_config(workspace)
    context = load_context("linux-x64", root=workspace)
    outside = tmp_path / "keep.txt"
    protected = workspace / ".pixi/env/keep.txt"
    artifact = workspace / "generated/remove.txt"
    for path in (outside, protected, artifact):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    clean.run(context, {"paths": [str(outside), str(workspace), str(protected)]})
    assert outside.exists() and protected.exists() and artifact.exists()
    clean.run(context, {"paths": [str(artifact.parent)]})
    assert not artifact.exists()
    assert outside.exists() and protected.exists()
