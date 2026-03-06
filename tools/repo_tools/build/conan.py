"""Conan, Emscripten, and Dawn helpers for the build tool."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

from repo_tools.core import ShellCommand, find_venv_executable, logger


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


def get_emsdk_version(root: Path) -> str:
    """Read the emsdk version from the local emsdk Conan recipe (single source of truth)."""
    emsdk_conanfile = root / "tools" / "conan" / "emsdk" / "conanfile.py"
    text = emsdk_conanfile.read_text()
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse emsdk version from {emsdk_conanfile}")
    return match.group(1)


def _write_emscripten_profile(profile_path: Path, emsdk_version: str) -> None:
    """Write a Conan profile for Emscripten cross-builds.

    Using a profile (rather than CLI -s:h/-c:h overrides) allows [tool_requires]
    to propagate emsdk and ninja to ALL dependency builds — not just the consumer.
    Without this, packages like OpenUSD fail to find emcc when building from source.
    """
    content = f"""\
[settings]
os=Emscripten
arch=wasm
compiler=emcc
compiler.version={emsdk_version}
compiler.cppstd=17
compiler.libcxx=libc++

[options]
*:shared=False

[tool_requires]
emsdk/{emsdk_version}
ninja/1.13.2

[conf]
tools.cmake.cmaketoolchain:generator=Ninja
tools.cmake.cmake_layout:build_folder_vars=["settings.os"]
tools.build:cflags=['-pthread']
tools.build:cxxflags=['-pthread', '-DTBB_USE_ASSERT=0', '-fexceptions']
tools.build:exelinkflags=['-pthread', '-fexceptions', '-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=4GB', '-sINITIAL_MEMORY=512MB']
tools.build:sharedlinkflags=['-pthread', '-fexceptions', '-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=4GB', '-sINITIAL_MEMORY=512MB']
"""
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(content, encoding="utf-8")


def get_emscripten_conan_flags(root: Path, build_folder: Path) -> list[str]:
    """Return Conan CLI flags for Emscripten cross-builds.

    Generates a host profile file in the build folder and returns flags that
    reference it.  All values are derived from the emsdk recipe (single source
    of truth).
    """
    v = get_emsdk_version(root)
    profile_path = build_folder / "conan_profile_emscripten"
    _write_emscripten_profile(profile_path, v)
    return [f"--profile:host={profile_path}"]


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


def ensure_emdawnwebgpu_port(root: Path, build_folder: Path) -> Path:
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
        with urllib.request.urlopen(url, timeout=60) as resp:
            tmp_file.write_bytes(resp.read())
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


def export_local_conan_recipes(root: Path, logs_dir: Path, conan_config: dict) -> None:
    """Export all local Conan recipes so they are available for dependency resolution."""
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
        ShellCommand([
            conan_exe,
            "export",
            str(recipe_dir),
            f"--name={name}",
            f"--version={version}",
        ]).exec(log_file=export_log_file)


def get_local_recipe_names(root: Path, conan_config: dict) -> set[str]:
    """Return the set of local recipe names from config."""
    recipes = _get_local_recipes(root, conan_config)
    names: set[str] = set()
    for recipe in recipes:
        if isinstance(recipe, dict) and recipe.get("name"):
            names.add(str(recipe["name"]))
    return names


def strip_local_recipe_revisions(
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


def load_conan_env(build_dir: Path, preset_type: str = "test") -> dict[str, str]:
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
