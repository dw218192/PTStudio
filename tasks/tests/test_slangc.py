"""Integration tests for pts_shaderc's generated outputs and dependency cache.

Run after building the native pts_shaderc target. Task-only environments skip
these checks when the host tool has not been built yet.
"""

import os
import subprocess
from pathlib import Path

import pytest
from tasks.build_support.conan import load_conan_env
from tasks.slangc import _resolve_pts_shaderc
from tasks.utils.project import load_context


@pytest.fixture(scope="module")
def shaderc():
    context = load_context()
    build_dir = Path(context.tokens["build_dir"])
    try:
        executable = _resolve_pts_shaderc(build_dir)
    except FileNotFoundError:
        pytest.skip("build native pts_shaderc before running compiler integration tests")
    environment = {**os.environ, **load_conan_env(executable.parent.parent)}

    def run(*args, success=True):
        result = subprocess.run(
            [str(executable), "compile", *map(str, args)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if success:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0
        return result

    return run


def test_cpp_regenerates_for_transitive_headers_missing_outputs_and_defines(shaderc, tmp_path):
    includes = tmp_path / "includes"
    includes.mkdir()
    constants = includes / "constants.h"
    constants.write_text("static const float gain = 2.0;\n", encoding="utf-8")
    helper = tmp_path / "helper.slang"
    helper.write_text(
        '#include "includes/constants.h"\nfloat apply(float x) { return gain * x; }\n',
        encoding="utf-8",
    )
    source = tmp_path / "exports.slang"
    source.write_text(
        "import helper;\nexport __extern_cpp float exported(float x) {\n"
        "#ifdef DOUBLE\nreturn apply(x) * 2;\n#else\nreturn apply(x);\n#endif\n}\n",
        encoding="utf-8",
    )
    output, header = tmp_path / "generated.cpp", tmp_path / "generated.h"
    args = ["--source", source, "--output", output, "--cpp-header", header]
    shaderc(*args)
    assert "up-to-date" in shaderc(*args).stdout
    before = output.read_text(encoding="utf-8")
    constants.write_text("static const float gain = 3.0;\n", encoding="utf-8")
    assert "up-to-date" not in shaderc(*args).stdout
    assert output.read_text(encoding="utf-8") != before
    assert "up-to-date" in shaderc(*args).stdout
    header.unlink()
    assert "up-to-date" not in shaderc(*args).stdout
    assert header.exists()
    before = output.read_text(encoding="utf-8")
    shaderc(*args, "-D", "DOUBLE")
    assert output.read_text(encoding="utf-8") != before
    assert "up-to-date" in shaderc(*args, "-D", "DOUBLE").stdout
    assert "up-to-date" not in shaderc(*args, "-D", "DOUBLE", "--force").stdout
    assert "up-to-date" in shaderc(*args, "-D", "DOUBLE").stdout


def test_upload_types_survive_binding_renames_and_reject_missing_names(shaderc, tmp_path):
    source = tmp_path / "upload.slang"
    source.write_text(
        "struct Data { float4 value; };\nConstantBuffer<Data> data;\n"
        '[shader("fragment")] float4 fs_main() : SV_Target0 { return data.value; }\n',
        encoding="utf-8",
    )
    output, header = tmp_path / "generated.wgsl", tmp_path / "upload.h"
    args = [
        "--source",
        source,
        "--output",
        output,
        "--types-header",
        header,
        "--types-namespace",
        "test_upload",
        "--type",
        "Data",
    ]
    shaderc(*args)
    header.unlink()
    shaderc(*args)
    generated = header.read_text(encoding="utf-8")
    assert "struct alignas(16) Data" in generated
    assert "up-to-date" in shaderc(*args).stdout
    source.write_text(
        source.read_text(encoding="utf-8").replace("data", "renamed_binding"),
        encoding="utf-8",
    )
    assert "up-to-date" not in shaderc(*args).stdout
    assert header.read_text(encoding="utf-8") == generated
    result = shaderc(*args[:-1], "missing", success=False)
    assert "type not found in reflected WGSL buffers: missing" in result.stderr
    # A failed compilation must not publish any of the new output bundle.
    assert "up-to-date" in shaderc(*args).stdout
