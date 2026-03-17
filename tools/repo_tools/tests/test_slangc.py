"""Tests for slangc dependency tracking."""

import os
import time
from pathlib import Path

import pytest

from repo_tools.slangc import _should_compile_shader


@pytest.fixture
def shader_dir(tmp_path: Path):
    """Create a temp directory with a .slang input and a compiled .wgsl output."""
    input_file = tmp_path / "main.slang"
    input_file.write_text("// main shader")

    output_file = tmp_path / "main.wgsl"
    output_file.write_text("// compiled")
    # Ensure output is newer than input
    _touch_newer(output_file, input_file)

    return tmp_path, input_file, output_file


def _touch_newer(target: Path, reference: Path) -> None:
    """Set target's mtime to be strictly newer than reference."""
    ref_mtime = reference.stat().st_mtime
    os.utime(target, (ref_mtime + 1, ref_mtime + 1))


class TestShouldCompileShader:
    def test_force_always_recompiles(self, shader_dir):
        _, input_file, output_file = shader_dir
        assert _should_compile_shader(input_file, output_file, force=True)

    def test_missing_output_recompiles(self, tmp_path):
        input_file = tmp_path / "main.slang"
        input_file.write_text("// shader")
        output_file = tmp_path / "main.wgsl"
        assert _should_compile_shader(input_file, output_file, force=False)

    def test_up_to_date_skips(self, shader_dir):
        _, input_file, output_file = shader_dir
        assert not _should_compile_shader(input_file, output_file, force=False)

    def test_input_newer_recompiles(self, shader_dir):
        _, input_file, output_file = shader_dir
        _touch_newer(input_file, output_file)
        assert _should_compile_shader(input_file, output_file, force=False)

    def test_sibling_slang_newer_recompiles(self, shader_dir):
        tmp_path, input_file, output_file = shader_dir
        # Create a sibling .slang file that is newer than the output
        sibling = tmp_path / "utils.slang"
        sibling.write_text("// utility module")
        _touch_newer(sibling, output_file)
        assert _should_compile_shader(input_file, output_file, force=False)

    def test_sibling_slang_older_skips(self, shader_dir):
        tmp_path, input_file, output_file = shader_dir
        # Create a sibling .slang file that is older than the output
        sibling = tmp_path / "utils.slang"
        sibling.write_text("// utility module")
        # Make sibling older than output
        out_mtime = output_file.stat().st_mtime
        os.utime(sibling, (out_mtime - 1, out_mtime - 1))
        assert not _should_compile_shader(input_file, output_file, force=False)

    def test_non_slang_sibling_ignored(self, shader_dir):
        tmp_path, input_file, output_file = shader_dir
        # A newer .txt file should NOT trigger recompilation
        other = tmp_path / "notes.txt"
        other.write_text("not a shader")
        _touch_newer(other, output_file)
        assert not _should_compile_shader(input_file, output_file, force=False)
