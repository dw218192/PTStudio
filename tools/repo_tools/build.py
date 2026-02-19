"""Build subcommand implementation (PTStudio project override)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    ToolContext,
    find_venv_executable,
    get_tool,
    invoke_tool,
    is_windows,
    log_section,
    logger,
    remove_tree_with_retries,
    run_command,
)


# ── Conan Profile ────────────────────────────────────────────────────


def ensure_conan_profile() -> None:
    """Ensure Conan profiles exist, run detect if needed."""
    profile_dir = Path.home() / ".conan2" / "profiles"

    if not profile_dir.exists() or not any(profile_dir.iterdir()):
        logger.info("No Conan profiles found. Running 'conan profile detect'...")
        conan_exe = find_venv_executable("conan")
        subprocess.run([conan_exe, "profile", "detect"], check=True)
    else:
        logger.info("Conan profiles already exist.")


# ── Emscripten Helpers ───────────────────────────────────────────────


def _get_emsdk_version(root: Path) -> str:
    """Read the emsdk version from the local emsdk Conan recipe (single source of truth)."""
    emsdk_conanfile = root / "tools" / "conan" / "emsdk" / "conanfile.py"
    text = emsdk_conanfile.read_text()
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse emsdk version from {emsdk_conanfile}")
    return match.group(1)


def _get_emscripten_conan_flags(root: Path) -> list[str]:
    """Return Conan CLI flags for Emscripten cross-builds.

    All values are derived from the emsdk recipe (single source of truth).
    These are passed as -s:h / -c:h overrides so no profile file is needed.
    """
    v = _get_emsdk_version(root)
    return [
        # Host settings
        "-s:h", "os=Emscripten",
        "-s:h", "arch=wasm",
        "-s:h", "compiler=emcc",
        "-s:h", f"compiler.version={v}",
        "-s:h", "compiler.cppstd=17",
        "-s:h", "compiler.libcxx=libc++",
        # Host conf -- build flags that propagate to all dependency builds
        "-c:h", "tools.cmake.cmaketoolchain:generator=Ninja",
        "-c:h", 'tools.cmake.cmake_layout:build_folder_vars=["settings.os"]',
        "-c:h", "tools.build:cflags=['-pthread']",
        "-c:h", "tools.build:cxxflags=['-pthread', '-DTBB_USE_ASSERT=0', '-fexceptions']",
        "-c:h", "tools.build:exelinkflags=['-pthread', '-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=4GB', '-sINITIAL_MEMORY=512MB']",
        "-c:h", "tools.build:sharedlinkflags=['-pthread', '-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=4GB', '-sINITIAL_MEMORY=512MB']",
    ]


# ── Dawn / emdawnwebgpu ──────────────────────────────────────────────


def _parse_conanfile_metadata(conanfile_path: Path) -> tuple[str | None, str | None]:
    """Extract name and version from a conanfile.py by parsing class attributes."""
    content = conanfile_path.read_text(encoding="utf-8")
    name_match = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    version_match = re.search(
        r'^\s*version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE
    )
    name = name_match.group(1) if name_match else None
    version = version_match.group(1) if version_match else None
    return name, version


def _get_dawn_version(root: Path) -> str:
    """Extract the Dawn version from the Dawn Conan recipe."""
    dawn_conanfile = root / "tools" / "conan" / "dawn" / "conanfile.py"
    _name, version = _parse_conanfile_metadata(dawn_conanfile)
    if not version:
        raise RuntimeError(f"Could not parse Dawn version from {dawn_conanfile}")
    return version


def _ensure_emdawnwebgpu_port(root: Path, build_folder: Path) -> Path:
    """Ensure the emdawnwebgpu remote port file matching Dawn is available.

    The emdawnwebgpu version is derived from the pinned Dawn version to keep
    native and WASM WebGPU APIs in sync. The remote port file is downloaded
    from Dawn's GitHub releases on demand into the build directory.

    Returns the path to the remote port file.
    """
    dawn_version = _get_dawn_version(root)
    tag = f"v{dawn_version}"
    filename = f"emdawnwebgpu-{tag}.remoteport.py"
    port_file = build_folder / filename

    if port_file.exists():
        logger.info(f"emdawnwebgpu port file found: {port_file.name}")
        return port_file

    # Download from Dawn releases into a temp file, then rename atomically
    # to avoid leaving a partial file on interrupted downloads.
    url = (
        f"https://github.com/google/dawn/releases/download/{tag}/{filename}"
    )
    logger.info(f"Downloading emdawnwebgpu port ({tag})...")
    build_folder.mkdir(parents=True, exist_ok=True)
    tmp_file = port_file.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_file)
        tmp_file.replace(port_file)
    except Exception as e:
        tmp_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download emdawnwebgpu port from {url}: {e}"
        ) from e

    logger.info(f"emdawnwebgpu port file saved: {port_file.name}")
    return port_file


# ── Conan Local Recipes ──────────────────────────────────────────────


def _discover_local_recipes(root: Path, recipes_dir: Path) -> list[dict]:
    """Discover all conan recipes in a directory by scanning for conanfile.py files."""
    recipes = []
    if not recipes_dir.exists():
        logger.warning(f"Local recipes directory does not exist: {recipes_dir}")
        return recipes

    for subdir in sorted(recipes_dir.iterdir()):
        conanfile = subdir / "conanfile.py"
        if not conanfile.exists():
            continue
        name, version = _parse_conanfile_metadata(conanfile)
        if not name or not version:
            logger.warning(
                f"Skipping {subdir.name}: could not parse name/version from conanfile.py"
            )
            continue
        recipes.append(
            {
                "name": name,
                "version": version,
                "path": str(subdir.relative_to(root)),
            }
        )
    return recipes


def _get_local_recipes(root: Path, conan_config: dict) -> list[dict]:
    """Get local recipes from config, supporting both directory and explicit list formats."""
    local_recipes = conan_config.get("local_recipes")
    if not local_recipes:
        return []

    # New format: string path to directory containing recipes
    if isinstance(local_recipes, str):
        recipes_dir = root / local_recipes
        return _discover_local_recipes(root, recipes_dir)

    # Old format: explicit list of recipe dicts
    if isinstance(local_recipes, list):
        return local_recipes

    logger.warning(f"Invalid local_recipes format: {type(local_recipes)}")
    return []


def _export_local_conan_recipes(root: Path, logs_dir: Path, conan_config: dict) -> None:
    recipes = _get_local_recipes(root, conan_config)
    if not recipes:
        return

    conan_exe = find_venv_executable("conan")
    for recipe in recipes:
        if not isinstance(recipe, dict):
            logger.warning(f"Skipping invalid recipe entry (not a dict): {recipe}")
            continue
        name = recipe.get("name")
        version = recipe.get("version")
        path_value = recipe.get("path")
        if not name or not version or not path_value:
            logger.warning(
                f"Skipping invalid recipe entry (missing name, version, or path): {recipe}"
            )
            continue
        recipe_dir = root / str(path_value)
        if not recipe_dir.exists():
            logger.warning(
                f"Skipping invalid recipe entry (path does not exist): {recipe}"
            )
            continue
        export_log_file = logs_dir / f"conan_export_{name}.log"
        run_command(
            [
                conan_exe,
                "export",
                str(recipe_dir),
                f"--name={name}",
                f"--version={version}",
            ],
            log_file=export_log_file,
        )


def _get_local_recipe_names(root: Path, conan_config: dict) -> set[str]:
    recipes = _get_local_recipes(root, conan_config)
    names: set[str] = set()
    for recipe in recipes:
        if isinstance(recipe, dict) and recipe.get("name"):
            names.add(str(recipe["name"]))
    return names


def _strip_local_recipe_revisions(
    lock_file: Path, local_recipe_names: set[str]
) -> None:
    """Strip revisions and timestamps from local recipe entries in the lock file.

    Local recipes are exported on each build, so their revisions are not stable.
    Removing revisions keeps the lock file stable while still pinning versions.
    """
    if not local_recipe_names:
        return

    if not lock_file.exists():
        return

    with open(lock_file, "r") as f:
        lock_data = json.load(f)

    modified = False
    for key in ["requires", "build_requires"]:
        if key not in lock_data:
            continue
        original = lock_data[key]
        updated = []
        for entry in original:
            if any(entry.startswith(f"{name}/") for name in local_recipe_names):
                no_timestamp = entry.split("%", 1)[0]
                no_revision = no_timestamp.split("#", 1)[0]
                updated.append(no_revision)
            else:
                updated.append(entry)
        if updated != original:
            lock_data[key] = updated
            modified = True

    if modified:
        with open(lock_file, "w") as f:
            json.dump(lock_data, f, indent=4)
        logger.info(
            "Stripped local recipe revisions in lock file: "
            f"{', '.join(sorted(local_recipe_names))}"
        )


# ── Conan Environment ────────────────────────────────────────────────


def _load_conan_env(build_dir: Path, preset_type: str = "test") -> dict[str, str]:
    """Load environment variables from Conan-generated CMakePresets.json."""
    presets_path = build_dir / "CMakePresets.json"
    if not presets_path.exists():
        logger.warning(f"CMakePresets.json not found: {presets_path}")
        return {}

    presets = json.loads(presets_path.read_text(encoding="utf-8"))

    key = {
        "test": "testPresets",
        "configure": "configurePresets",
        "build": "buildPresets",
    }.get(preset_type, f"{preset_type}Presets")

    preset_list = presets.get(key, [])
    if not preset_list:
        logger.warning(f"No {key} found in {presets_path}")
        return {}

    env = preset_list[0].get("environment", {})
    if not env:
        return {}

    resolved: dict[str, str] = {}
    for var, value in env.items():
        if not isinstance(value, str):
            continue
        result = re.sub(
            r"\$penv\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
        resolved[var] = result
    return resolved


# ── Prebuild / Postbuild Steps ───────────────────────────────────────


def execute_build_steps(
    root: Path,
    config: dict,
    tokens: dict[str, str],
    dimensions: dict[str, str],
    logs_dir: Path,
    steps_config: dict,
    step_type: str,
    current_tool: str,
) -> None:
    """Execute prebuild or postbuild steps defined in config.

    Args:
        root: Repository root path
        config: Full configuration dictionary
        tokens: Token dictionary for invoke_tool
        dimensions: Dimension values for invoke_tool
        logs_dir: Logs directory path
        steps_config: Dictionary of build steps from config
        step_type: Either "prebuild" or "postbuild" for logging
        current_tool: Name of the current tool (to prevent recursion)
    """
    if not steps_config:
        return

    for step_name, step_config in steps_config.items():
        if not isinstance(step_config, dict):
            logger.warning(
                f"Skipping invalid {step_type} step '{step_name}': not a dict"
            )
            continue

        repo_tool = step_config.get("repo_tool", step_name)
        if not repo_tool:
            logger.warning(
                f"Skipping {step_type} step '{step_name}': missing 'repo_tool'"
            )
            continue
        step_args_value = step_config.get("args")
        if step_args_value is None:
            step_args_value = {
                key: value for key, value in step_config.items() if key != "repo_tool"
            }

        tool = get_tool(repo_tool)
        if tool is None:
            logger.error(f"  ✗ Unknown repo tool: {repo_tool}")
            raise RuntimeError(
                f"Unknown repo tool '{repo_tool}' in {step_type} step '{step_name}'"
            )
        if repo_tool == current_tool:
            logger.error(
                f"  ✗ Cannot call '{repo_tool}' tool from {step_type} steps (would cause recursion)"
            )
            raise RuntimeError(
                f"{step_type} step '{step_name}' cannot use '{repo_tool}' tool"
            )

        logger.info(f"Running {step_type} step: {step_name} (tool: {repo_tool})")

        try:
            invoke_tool(repo_tool, tokens, config, dimensions=dimensions, extra_args=step_args_value)
            logger.info(f"  ✓ {step_name} completed")
        except Exception as e:
            logger.error(f"  ✗ {step_name} failed: {e}")
            raise RuntimeError(f"{step_type} step '{step_name}' failed") from e


# ── IDE Generation ───────────────────────────────────────────────────


def _format_workspace_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
        return f"${{workspaceFolder}}/{relative.as_posix()}"
    except ValueError:
        return path.as_posix()


def _ensure_cmake_file_api_query(build_dir: Path) -> None:
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
                target_json = json.loads(
                    (reply_dir / target["jsonFile"]).read_text(encoding="utf-8")
                )
                if not _target_has_plugin_sources(target_json, plugins_root):
                    target_json = None
            if target_json is None:
                continue
            for group in target_json.get("compileGroups", []):
                for include in group.get("includes", []):
                    include_dirs.add(Path(include["path"]))
                for define in group.get("defines", []):
                    defines.add(define["define"])
    return include_dirs, defines


def _generate_cpp_properties(root: Path, build_dir: Path, windowing: str) -> None:
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
    browse_paths = include_paths

    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": "PTStudio",
        "cppStandard": "c++17",
        "cStandard": "c17",
        "defines": sorted(defines),
        "includePath": include_paths,
        "browse": {"path": browse_paths},
    }
    payload = {"version": 4, "configurations": [config]}
    (vscode_dir / "c_cpp_properties.json").write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8"
    )


def _generate_launch_json(
    root: Path,
    build_dir: Path,
    build_type: str,
    test_names: list[str],
    env_vars: dict,
) -> None:
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
    return target_name.startswith(("test_", "test"))


def _discover_test_targets(build_dir: Path) -> list[str]:
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


# ── Arg Helpers ──────────────────────────────────────────────────────


def _get_dict_arg(args: dict[str, Any], field_name: str) -> dict:
    """Extract a dict argument from args, warn if non-dict, return {} if None or invalid."""
    value = args.get(field_name, {})
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    logger.warning(f"Build arg '{field_name}' must be a dict; ignoring.")
    return {}


# ── Main Build Logic ─────────────────────────────────────────────────


def build_command(ctx: ToolContext, args: dict[str, Any], current_tool: str) -> None:
    """Meta-meta-build system implementation.

    1. Configure the project
       - fetch dependencies with conan
       - configure CMake
       - generate vscode cpp include paths for the project
    2. Build the project using CMake
    3. Generate vscode launch configurations for the project
    """
    root = ctx.workspace_root
    config = ctx.config
    tokens = ctx.tokens
    dimensions = ctx.dimensions

    # Platform is already determined by the framework
    platform_id = dimensions.get("platform", "")
    build_type = dimensions.get("build_type", "Debug")

    # Derive paths from the token system (resolved by the framework)
    build_root = Path(tokens.get("build_root", str(root / "_build")))
    build_folder = build_root / platform_id
    build_dir = Path(tokens.get("build_dir", str(build_folder / build_type)))
    logs_dir = Path(tokens.get("logs_root", str(root / "_logs")))
    windowing = args.get("windowing", "glfw")

    # Conan-specific paths from tokens
    conan_deps_root = Path(tokens.get("conan_deps_root", str(build_root / platform_id / "deps")))
    conan_lock_name = tokens.get("conan_lock", "conan_glfw.lock")
    lock_file = root / conan_lock_name

    conan_config = _get_dict_arg(args, "conan")
    prebuild_steps = _get_dict_arg(args, "prebuild")
    postbuild_steps = _get_dict_arg(args, "postbuild")

    conan_profile = args.get("conan_profile", "default")

    # Emscripten build configuration
    emscripten_build = platform_id == "emscripten"
    if emscripten_build:
        # Use separate lock file for Emscripten builds
        lock_file = root / "conan_emscripten.lock"
        logger.info("Emscripten build mode: cross-building via Conan")
        logger.info(f"Lock file: {lock_file}")

    # Remove build configuration directory if -x flag is provided
    if args.get("rebuild") and build_dir.exists():
        logger.info(f"Rebuild flag (-x) detected. Removing build directory: {build_dir}")
        remove_tree_with_retries(build_dir)

    # Create build directory if missing
    build_folder.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Change to build directory
    original_cwd = os.getcwd()
    try:
        os.chdir(build_folder)

        if args.get("build_only"):
            logger.info("Build only mode (-b): Skipping configuration steps")
            logger.info(f"Building with configuration: {build_type}")
        else:
            ensure_conan_profile()
            _export_local_conan_recipes(root, logs_dir, conan_config)

            if args.get("configure_only"):
                logger.info(f"Configuring with configuration: {build_type}")
            else:
                logger.info(f"Building with configuration: {build_type}")

            # Handle lock file generation and usage
            should_create_lock = args.get("update_lock") or not lock_file.exists()

            # Emscripten flags override the default host profile settings/conf
            emscripten_flags = _get_emscripten_conan_flags(root) if emscripten_build else []

            with log_section("Conan dependencies"):
                local_recipe_names = _get_local_recipe_names(root, conan_config)
                if should_create_lock:
                    if args.get("update_lock"):
                        logger.info(
                            "Update lock flag (-u) detected. Regenerating lock file..."
                        )
                    else:
                        logger.info("Lock file not found. Generating new lock file...")
                    lock_log_file = logs_dir / f"conan_lock_create_{windowing}.log"
                    conan_exe = find_venv_executable("conan")
                    lock_args = [
                        conan_exe,
                        "lock",
                        "create",
                        str(root),
                        "-o",
                        f"&:windowing={windowing}",
                        "--lockfile-out",
                        str(lock_file),
                        f"--profile:host={conan_profile}",
                        f"--profile:build={conan_profile}",
                        *emscripten_flags,
                    ]
                    run_command(lock_args, log_file=lock_log_file)
                    # Strip revisions for local recipes in the lock file
                    _strip_local_recipe_revisions(lock_file, local_recipe_names)
                else:
                    logger.info(f"Lock file found. Using existing lock file: {lock_file}")

                install_log_file = logs_dir / "conan_install.log"
                conan_exe = find_venv_executable("conan")

                logger.info("Installing dependencies with Conan...")
                run_command(
                    [
                        conan_exe,
                        "install",
                        str(root),
                        "--lockfile",
                        str(lock_file),
                        f"--output-folder={build_type}",
                        f"--deployer-folder={conan_deps_root}",
                        "--deployer=full_deploy",
                        "--deployer=runtime_deploy",
                        "--build=missing",
                        f"--profile:host={conan_profile}",
                        f"--profile:build={conan_profile}",
                        "-o",
                        f"&:windowing={windowing}",
                        "-s",
                        "compiler.cppstd=17",
                        "-s",
                        f"build_type={build_type}",
                        *emscripten_flags,
                    ],
                    log_file=install_log_file,
                )

            # Execute prebuild steps
            if prebuild_steps:
                with log_section("Prebuild steps"):
                    execute_build_steps(
                        root,
                        config,
                        tokens,
                        dimensions,
                        logs_dir,
                        prebuild_steps,
                        "prebuild",
                        current_tool,
                    )

            # Conan build environment script (vcvars on Windows for Ninja + MSVC)
            conanbuild = build_dir / "conanbuild"

            with log_section("CMake configure"):
                configure_log_file = logs_dir / "cmake_configure.log"
                cmake_exe = find_venv_executable("cmake")
                _ensure_cmake_file_api_query(build_folder / build_type)

                bt = build_type.lower()
                if emscripten_build:
                    preset_name = f"conan-emscripten-{bt}"
                else:
                    preset_name = f"conan-{bt}"

                cmake_args = [
                    cmake_exe,
                    "--preset",
                    preset_name,
                    "-S",
                    str(root),
                ]
                if emscripten_build:
                    emdawnwebgpu_port = _ensure_emdawnwebgpu_port(root, build_folder)
                    cmake_args.append(f"-DEMDAWNWEBGPU_PORT_FILE={emdawnwebgpu_port}")

                run_command(cmake_args, log_file=configure_log_file, env_script=conanbuild)

            _generate_cpp_properties(root, build_dir, windowing)

            if not args.get("configure_only"):
                with log_section("CMake build"):
                    build_log_file = logs_dir / "cmake_build.log"
                    cmake_exe = find_venv_executable("cmake")

                    bt = build_type.lower()
                    if emscripten_build:
                        preset_name = f"conan-emscripten-{bt}"
                    else:
                        preset_name = f"conan-{bt}"

                    build_args = [cmake_exe, "--build", "--preset", preset_name]
                    # Build presets require CMakeUserPresets.json at project root
                    os.chdir(root)
                    try:
                        run_command(build_args, log_file=build_log_file, env_script=conanbuild)
                    finally:
                        os.chdir(build_folder)

                # Execute postbuild steps
                if postbuild_steps:
                    with log_section("Postbuild steps"):
                        execute_build_steps(
                            root,
                            config,
                            tokens,
                            dimensions,
                            logs_dir,
                            postbuild_steps,
                            "postbuild",
                            current_tool,
                        )
            else:
                logger.info("Configure only mode (-c): Skipping build step")

        tests = _discover_test_targets(build_dir)
        env_vars = _load_conan_env(build_dir, preset_type="test")
        _generate_launch_json(root, build_dir, build_type, tests, env_vars)
    finally:
        os.chdir(original_cwd)


# ── Tool Class ───────────────────────────────────────────────────────


class BuildTool(RepoTool):
    name = "build"
    help = "Build the project"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.option(
            "-x", "--rebuild",
            is_flag=True,
            default=None,
            help=(
                "Rebuild flag: removes build configuration folder before building "
                "(Use clean tool for a full clean of build and dependencies folders)"
            ),
        )(cmd)
        cmd = click.option(
            "-u", "--update-lock",
            is_flag=True,
            default=None,
            help="Update lock flag: forces regeneration of conan.lock",
        )(cmd)
        cmd = click.option(
            "-c", "--configure-only",
            is_flag=True,
            default=None,
            help=(
                "Configure only flag: runs conan install and cmake configure, "
                "but skips building"
            ),
        )(cmd)
        cmd = click.option(
            "-b", "--build-only",
            is_flag=True,
            default=None,
            help=(
                "Build only flag: skips conan install and cmake configure, "
                "only runs build"
            ),
        )(cmd)
        cmd = click.option(
            "--conan-profile",
            default=None,
            help="Conan profile (default: default)",
        )(cmd)
        cmd = click.option(
            "--windowing",
            type=click.Choice(["glfw", "null"]),
            default=None,
            help="Windowing backend (default: glfw)",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "rebuild": False,
            "update_lock": False,
            "configure_only": False,
            "build_only": False,
            "conan_profile": "default",
            "windowing": "glfw",
            "prebuild": {},
            "postbuild": {},
            "conan": {},
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        build_command(ctx, args, self.name)
