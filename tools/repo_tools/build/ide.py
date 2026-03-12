"""VS Code generation and test discovery (CMake File API)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from repo_tools.core import is_windows


# ── CMake File API Helpers ────────────────────────────────────────────


def _format_workspace_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return path.as_posix()


def ensure_cmake_file_api_query(build_dir: Path) -> None:
    """Create the CMake File API query file so codemodel data is generated."""
    query_dir = build_dir / ".cmake" / "api" / "v1" / "query"
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / "codemodel-v2").write_text("", encoding="utf-8")


def _load_codemodel(build_dir: Path) -> tuple[dict, Path] | None:
    reply_dir = build_dir / ".cmake" / "api" / "v1" / "reply"
    if not reply_dir.exists():
        return None
    index_files = sorted(
        reply_dir.glob("index-*.json"), key=lambda p: p.stat().st_mtime
    )
    if not index_files:
        return None
    index = json.loads(index_files[-1].read_text(encoding="utf-8"))
    codemodel_info = index.get("reply", {}).get("codemodel-v2")
    if not codemodel_info:
        return None
    codemodel = json.loads(
        (reply_dir / codemodel_info["jsonFile"]).read_text(encoding="utf-8")
    )
    return codemodel, reply_dir


# ── Target Analysis ──────────────────────────────────────────────────


def _target_has_plugin_sources(target_json: dict, plugins_root: Path) -> bool:
    for source in target_json.get("sources", []):
        source_path = Path(source.get("path", ""))
        try:
            source_path.relative_to(plugins_root)
            return True
        except ValueError:
            continue
    return False


def _collect_target_compile_info(
    codemodel: dict,
    reply_dir: Path,
    root: Path,
    target_names: set[str] | None,
    include_plugins: bool,
) -> tuple[set[Path], set[str]]:
    include_dirs: set[Path] = set()
    defines: set[str] = set()
    plugins_root = root / "plugins"
    include_all = not target_names
    for config in codemodel.get("configurations", []):
        for target in config.get("targets", []):
            target_name = target.get("name")
            if not target_name:
                continue
            target_json = None
            if include_all or target_name in target_names:
                target_json = json.loads(
                    (reply_dir / target["jsonFile"]).read_text(encoding="utf-8")
                )
            elif include_plugins:
                candidate = json.loads(
                    (reply_dir / target["jsonFile"]).read_text(encoding="utf-8")
                )
                if _target_has_plugin_sources(candidate, plugins_root):
                    target_json = candidate
            if target_json is None:
                continue
            for group in target_json.get("compileGroups", []):
                for include in group.get("includes", []):
                    include_dirs.add(Path(include["path"]))
                for define in group.get("defines", []):
                    defines.add(define["define"])
    return include_dirs, defines


# ── VS Code Generation ───────────────────────────────────────────────


def _detect_compiler_path(build_dir: Path) -> str | None:
    """Try to find the C++ compiler path from CMakeCache.txt."""
    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.exists():
        return "cl.exe" if is_windows() else None
    text = cache_file.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:"):
            _, _, value = line.partition("=")
            return value.strip()
    return "cl.exe" if is_windows() else None


def generate_cpp_properties(root: Path, build_dir: Path, windowing: str) -> None:
    """Generate .vscode/c_cpp_properties.json from CMake File API data."""
    # Use CMake file-api outputs for accuracy across generators/platforms.
    # compile_commands.json is not reliably generated on all platforms or generators.
    result = _load_codemodel(build_dir)
    include_dirs: set[Path] = set()
    defines: set[str] = set()
    if result:
        codemodel, reply_dir = result
        include_dirs, defines = _collect_target_compile_info(
            codemodel,
            reply_dir,
            root,
            None,
            include_plugins=True,
        )

    if not defines:
        defines = {
            "SPDLOG_FMT_EXTERNAL",
            f"PTS_WINDOWING_{windowing}",
            f'PTS_WINDOWING="{windowing}"',
        }

    include_paths = sorted(
        {_format_workspace_path(root, path) for path in include_dirs if path.exists()}
    )

    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    compiler_path = _detect_compiler_path(build_dir)

    config = {
        "name": "PTStudio",
        "cppStandard": "c++17",
        "cStandard": "c17",
        "defines": sorted(defines),
        "includePath": include_paths,
    }
    if compiler_path:
        config["compilerPath"] = compiler_path
    if is_windows():
        config["intelliSenseMode"] = "msvc-x64"
    else:
        config["intelliSenseMode"] = "gcc-x64"

    payload = {"version": 4, "configurations": [config]}
    (vscode_dir / "c_cpp_properties.json").write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8"
    )


def _find_renderdoc() -> Path | None:
    """Find RenderDoc UI executable on this host."""
    import shutil

    rdoc = shutil.which("qrenderdoc")
    if rdoc:
        return Path(rdoc)

    if not is_windows():
        return None

    candidate = Path("C:/Program Files/RenderDoc/qrenderdoc.exe")
    return candidate if candidate.exists() else None


def _find_nsight_graphics() -> Path | None:
    """Find Nsight Graphics CLI executable on this host."""
    import shutil

    ngfx = shutil.which("ngfx")
    if ngfx:
        return Path(ngfx)

    if not is_windows():
        return None

    base = Path("C:/Program Files/NVIDIA Corporation")
    if not base.exists():
        return None
    candidates = sorted(
        base.glob("Nsight Graphics */host/windows-desktop-nomad-x64/ngfx-capture.exe"),
        reverse=True,
    )
    return candidates[0] if candidates else None


def generate_launch_json(
    root: Path,
    build_dir: Path,
    build_type: str,
    test_names: list[str],
    env_vars: dict,
) -> None:
    """Generate or update .vscode/launch.json with editor and test launch configs."""
    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    launch_path = vscode_dir / "launch.json"
    existing = {}
    if launch_path.exists():
        try:
            existing = json.loads(launch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    configurations = existing.get("configurations", [])
    if not isinstance(configurations, list):
        configurations = []

    path_separator = ";" if is_windows() else ":"
    env_entries = []
    for key, value in env_vars.items():
        if key.upper() == "PATH":
            env_entries.append(
                {"name": key, "value": f"{value}{path_separator}${{env:PATH}}"}
            )
        else:
            env_entries.append({"name": key, "value": value})

    launch_configs = []
    editor_name = f"PTStudio Editor ({build_type})"
    editor_path = build_dir / "bin" / ("editor.exe" if is_windows() else "editor")
    editor_program = _format_workspace_path(root, editor_path)

    if is_windows():
        launch_configs.append(
            {
                "name": editor_name,
                "type": "cppvsdbg",
                "request": "launch",
                "program": editor_program,
                "args": [],
                "cwd": "${workspaceFolder}",
                "console": "integratedTerminal",
                "environment": env_entries,
            }
        )

        # RenderDoc frame capture (F12 or PrintScreen to capture)
        renderdoc_path = _find_renderdoc()
        if renderdoc_path:
            # Generate a .cap file so RenderDoc launches the app with
            # the correct working directory and environment.
            cap_file = root / "_build" / "editor.cap"
            path_value = env_vars.get("PATH", "")
            env_str = ""
            if path_value:
                env_str = f"PATH={path_value}"
            cap_content = {
                "rdocCaptureSettings": 1,
                "settings": {
                    "autoConnect": True,
                    "commandLine": "",
                    "environment": [
                        e for e in [
                            {"separator": "Platform style",
                             "type": "Prepend",
                             "variable": "PATH",
                             "value": path_value} if path_value else None,
                            {"separator": "Platform style",
                             "type": "Set",
                             "variable": "PTSTUDIO_GPU_BACKEND",
                             "value": "Vulkan"},
                        ] if e is not None
                    ],
                    "executable": str(editor_path),
                    "inject": False,
                    "numQueuedFrames": 0,
                    "options": {
                        "allowFullscreen": True,
                        "allowVSync": True,
                        "apiValidation": True,
                        "captureAllCmdLists": True,
                        "captureCallstacks": False,
                        "debugOutputMute": True,
                        "delayForDebugger": 0,
                        "hookIntoChildren": False,
                        "refAllResources": False,
                        "verifyBufferAccess": False,
                    },
                    "queuedFrameCap": 0,
                    "workingDir": str(root),
                },
            }
            cap_file.parent.mkdir(parents=True, exist_ok=True)
            cap_file.write_text(
                json.dumps(cap_content, indent=4) + "\n", encoding="utf-8"
            )

            launch_configs.append(
                {
                    "name": "PTStudio Editor (RenderDoc)",
                    "type": "cppvsdbg",
                    "request": "launch",
                    "program": str(renderdoc_path),
                    "args": [str(cap_file)],
                    "cwd": "${workspaceFolder}",
                    "console": "integratedTerminal",
                }
            )

        # Nsight Graphics frame debugger
        nsight_path = _find_nsight_graphics()
        if nsight_path:
            nsight_env = list(env_entries) + [
                {"name": "PTSTUDIO_GPU_BACKEND", "value": "Vulkan"},
            ]
            launch_configs.append(
                {
                    "name": "PTStudio Editor (Nsight Graphics)",
                    "type": "cppvsdbg",
                    "request": "launch",
                    "program": str(nsight_path),
                    "args": [
                        f"--exe={editor_path}",
                        "--capture-hotkey=F11",
                    ],
                    "cwd": "${workspaceFolder}",
                    "console": "integratedTerminal",
                    "environment": nsight_env,
                }
            )

    for test_name in test_names:
        if is_windows():
            test_path = build_dir / "bin" / "tests" / f"{test_name}.exe"
            launch_configs.append(
                {
                    "name": f"PTStudio Test {test_name} ({build_type})",
                    "type": "cppvsdbg",
                    "request": "launch",
                    "program": _format_workspace_path(root, test_path),
                    "args": [],
                    "cwd": "${workspaceFolder}",
                    "console": "integratedTerminal",
                    "environment": env_entries,
                }
            )

    names_to_replace = {config["name"] for config in launch_configs}
    configurations = [
        config
        for config in configurations
        if config.get("name") not in names_to_replace
    ]
    configurations.extend(launch_configs)

    payload = {
        "version": existing.get("version", "0.2.0"),
        "configurations": configurations,
    }
    launch_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


# ── Test Target Discovery ────────────────────────────────────────────


def _is_test_name(target_name: str) -> bool:
    # Match testCamelCase or test_snake_case, but not e.g. "testing_utils"
    if not target_name.startswith("test"):
        return False
    rest = target_name[4:]
    return not rest or rest[0].isupper() or rest[0] == "_"


def discover_test_targets(build_dir: Path) -> list[str]:
    """Discover test executables from the CMake File API codemodel."""
    tests_dir = build_dir / "bin" / "tests"
    test_names: set[str] = set()
    result = _load_codemodel(build_dir)
    if result:
        codemodel, reply_dir = result
        for config in codemodel.get("configurations", []):
            for target in config.get("targets", []):
                target_json_path = reply_dir / target.get("jsonFile", "")
                if not target_json_path.exists():
                    continue
                target_json = json.loads(target_json_path.read_text(encoding="utf-8"))
                for artifact in target_json.get("artifacts", []):
                    artifact_path_str = artifact.get("path", "")
                    if not artifact_path_str:
                        continue
                    artifact_path = Path(artifact_path_str)
                    if not artifact_path.is_absolute():
                        artifact_path = (build_dir / artifact_path).resolve()
                    try:
                        artifact_path.relative_to(tests_dir)
                    except ValueError:
                        continue
                    if is_windows():
                        if artifact_path.suffix.lower() != ".exe":
                            continue
                        test_name = artifact_path.stem
                    else:
                        if not os.access(artifact_path, os.X_OK):
                            continue
                        test_name = artifact_path.name
                    if _is_test_name(test_name):
                        test_names.add(test_name)

    return sorted(test_names)
