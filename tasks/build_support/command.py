"""Build orchestrator -- main build_command and helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from tasks.utils import (
    CommandGroup,
    ProjectContext,
    find_executable,
    logger,
    remove_tree_with_retries,
    sanitized_subprocess_env,
)

from .conan import (
    ensure_conan_profile,
    ensure_emdawnwebgpu_port,
    export_local_conan_recipes,
    get_emscripten_conan_flags,
    get_local_recipe_names,
    load_conan_env,
    strip_local_recipe_revisions,
)
from .ide import (
    discover_test_targets,
    ensure_cmake_file_api_query,
    generate_cpp_properties,
    generate_launch_json,
)


def prepare_assets(ctx: ProjectContext, *, host_only: bool = False) -> None:
    """Run the project's asset generation in dependency order."""
    from tasks import embed, fmt, shader_variants_codegen, slangc, usdz

    if host_only:
        usdz.run(ctx, {})
        slangc.run(ctx, {})
        return
    fmt.run(ctx, {})
    slangc.run(ctx, {})
    shader_variants_codegen.run(ctx, {})
    usdz.run(ctx, {})
    embed.run(ctx, {})


# -- Helpers ----------------------------------------------------------


def _host_package_names(lock_file: Path) -> list[str]:
    """Read a Conan lock file and return the host requirement package names.

    This excludes build_requires (b2, cmake, emsdk, etc.) so that ``--build=``
    flags only force-rebuild host libraries, not build tools.
    """
    with open(lock_file) as f:
        lock = json.load(f)
    names = []
    for ref in lock.get("requires", []):
        # ref format: "name/version#revision%timestamp" or "name/version"
        name = ref.split("/", 1)[0]
        names.append(name)
    return names


# Prebuild tools that require a compiled host binary (two-phase build).
# Each entry maps a prebuild step name to a descriptor:
#   - target: CMake target name (also used as the built executable basename)
#   - dir:    Path (from repo root) of a standalone CMake project with a
#             `conanfile.txt` for its minimum dep set and a `CMakeLists.txt`
#             that builds the target.
_HOST_TOOL_TARGETS: dict[str, dict[str, str]] = {
    "usdz": {"target": "usdz_pack", "dir": "tools/conan/usdz_pack"},
    "slangc": {"target": "pts_shaderc", "dir": "tools"},
}


def _cmake_preset_name(build_type: str, emscripten: bool) -> str:
    bt = build_type.lower()
    return f"conan-emscripten-{bt}" if emscripten else f"conan-{bt}"


def _deploy_is_current(lock_file: Path, conan_deps_root: Path, build_type: str) -> bool:
    """Check if the deploy folder is already up-to-date with the lock file.

    Compares a hash of the lock file contents + build_type against a sentinel
    stored in the deploy folder.  Returns True when they match (deploy can be
    skipped), False otherwise.
    """
    sentinel = conan_deps_root / ".deploy_hash"
    if not sentinel.exists() or not conan_deps_root.exists():
        return False
    h = hashlib.sha256()
    h.update(lock_file.read_bytes())
    h.update(build_type.encode())
    return sentinel.read_text().strip() == h.hexdigest()


def _write_deploy_sentinel(lock_file: Path, conan_deps_root: Path, build_type: str) -> None:
    """Write the deploy sentinel after a successful deploy."""
    conan_deps_root.mkdir(parents=True, exist_ok=True)
    sentinel = conan_deps_root / ".deploy_hash"
    h = hashlib.sha256()
    h.update(lock_file.read_bytes())
    h.update(build_type.encode())
    sentinel.write_text(h.hexdigest())


# -- Host-tools-only Build --------------------------------------------


def _host_tools_only_build(
    root: Path,
    build_dir: Path,
    build_folder: Path,
    conan_deps_root: Path,
    logs_dir: Path,
    build_type: str,
    conan_profile: str,
    conan_config: dict,
    ctx: ProjectContext,
    build_env: dict,
) -> None:
    """Build host tools standalone without the root project Conan graph.

    Each host tool lives in its own directory with a `conanfile.txt` (minimum
    dep set) and a `CMakeLists.txt`. For each tool:
        conan install <dir> -of <tool_out>     # resolve deps
        cmake -S <dir> -B <tool_out>/cmake-build ...
        cmake --build ...
        copy <built exe> to {build_dir}/bin/

    Then runs only the prebuild steps mapped to _HOST_TOOL_TARGETS.
    """
    ensure_conan_profile()
    export_local_conan_recipes(root, logs_dir, conan_config)

    conan_exe = find_executable("conan")
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    is_win = sys.platform == "win32"

    with CommandGroup("Host tools (isolated)", cwd=build_folder, env=build_env) as g:
        for prebuild_name, spec in _HOST_TOOL_TARGETS.items():
            target_name = spec["target"]
            tool_dir = root / spec["dir"]
            exe_name = f"{target_name}.exe" if is_win else target_name
            dest = bin_dir / exe_name

            conanfile_txt = tool_dir / "conanfile.txt"
            if not conanfile_txt.exists():
                raise RuntimeError(
                    f"Host tool dep manifest not found: {conanfile_txt} "
                    f"(required for _HOST_TOOL_TARGETS entry '{prebuild_name}')"
                )
            tool_out = build_folder / "host_tools" / target_name
            tool_out.mkdir(parents=True, exist_ok=True)

            g.run(
                [
                    conan_exe,
                    "install",
                    str(tool_dir),
                    "--build=missing",
                    f"--output-folder={tool_out}",
                    f"--profile:host={conan_profile}",
                    f"--profile:build={conan_profile}",
                    "-s",
                    "compiler.cppstd=17",
                    "-s",
                    f"build_type={build_type}",
                ],
                log_file=logs_dir / f"conan_install_{target_name}.log",
            )

            # CMakeToolchain writes conan_toolchain.cmake under the generators
            # subfolder (multi-config layout) or at the top (single-config).
            toolchain = tool_out / "build" / "generators" / "conan_toolchain.cmake"
            if not toolchain.exists():
                toolchain = tool_out / "conan_toolchain.cmake"
            if not toolchain.exists():
                hits = list(tool_out.rglob("conan_toolchain.cmake"))
                if not hits:
                    raise RuntimeError(f"conan_toolchain.cmake not generated under {tool_out}")
                toolchain = hits[0]

            cmake_build = tool_out / "cmake-build"
            g.run(
                [
                    "cmake",
                    "-S",
                    str(tool_dir),
                    "-B",
                    str(cmake_build),
                    f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                ],
                log_file=logs_dir / f"cmake_configure_{target_name}.log",
            )
            g.run(
                [
                    "cmake",
                    "--build",
                    str(cmake_build),
                    "--target",
                    target_name,
                    "--config",
                    build_type,
                ],
                log_file=logs_dir / f"cmake_build_{target_name}.log",
            )

            built: Path | None = None
            for candidate in cmake_build.rglob(exe_name):
                if candidate.is_file() and candidate.stat().st_size > 0:
                    built = candidate
                    break
            if built is None:
                raise RuntimeError(f"Built host tool '{exe_name}' not found under {cmake_build}")
            shutil.copy2(built, dest)
            logger.info(f"Staged host tool: {dest} (from {built})")

            # Stage the tool's conanrun script alongside the binary so the
            # caller (e.g. slangc prebuild step) can source the tool's
            # isolated runtime env to find its dynamic deps -- critical on
            # Windows where DLLs are found via PATH. Each host tool has its
            # own isolated Conan graph, so its conanrun is specific to its
            # dependency set. Conan's layout can place it at either the
            # top of tool_out or under `build/generators/`.
            conanrun_name = "conanrun.bat" if is_win else "conanrun.sh"
            for loc in (
                tool_out / conanrun_name,
                tool_out / "build" / "generators" / conanrun_name,
            ):
                if loc.exists():
                    shutil.copy2(loc, bin_dir / conanrun_name)
                    # Also stage companion files (activate/deactivate, env .sh/.bat)
                    # so sourcing conanrun works fully.
                    for companion in loc.parent.glob("conan*"):
                        if companion.is_file():
                            shutil.copy2(companion, bin_dir / companion.name)
                    break

    with CommandGroup("Generate assets with host tools"):
        prepare_assets(ctx, host_only=True)

    logger.info("Host-tools-only build complete")


# -- Main Build Logic -------------------------------------------------


def build_command(ctx: ProjectContext, args: dict[str, Any]) -> None:
    """Configure dependencies, generate assets, and build with CMake.

    1. Configure the project
       - fetch dependencies with conan
       - configure CMake
       - generate vscode cpp include paths for the project
    2. Build the project using CMake
    3. Generate vscode launch configurations for the project
    """
    root = ctx.workspace_root
    tokens = ctx.tokens
    dimensions = ctx.dimensions

    # Build selectors come from this command's CLI options.
    platform_id = dimensions.get("platform", "")
    build_type = dimensions.get("build_type", "Debug")

    # Paths are expanded from the project configuration.
    build_root = Path(tokens["build_root"])
    build_folder = build_root / platform_id
    build_dir = Path(tokens["build_dir"])
    logs_dir = Path(tokens["logs_root"])
    windowing = args.get("windowing", "glfw")
    usd_modules = args.get("usd_modules") or []

    # Conan-specific paths from tokens
    conan_deps_root = Path(tokens["conan_deps_root"])

    conan_config = args.get("conan") or {}

    conan_profile = args.get("conan_profile", "default")

    # Emscripten build configuration
    emscripten_build = platform_id == "emscripten"
    host_tools_only = bool(args.get("host_tools_only"))
    if host_tools_only and emscripten_build:
        raise RuntimeError(
            "--host-tools-only requires a native platform; "
            "refusing to run with --platform emscripten"
        )

    # Host-tools-only short-circuits before touching the root project's
    # Conan graph -- the root lock file isn't cross-platform (e.g. Linux
    # GLFW pulls in xorg/system not present in conan_glfw.lock).
    if host_tools_only:
        logs_dir.mkdir(parents=True, exist_ok=True)
        build_folder.mkdir(parents=True, exist_ok=True)
        build_env = sanitized_subprocess_env()
        _host_tools_only_build(
            root,
            build_dir,
            build_folder,
            conan_deps_root,
            logs_dir,
            build_type,
            conan_profile,
            conan_config,
            ctx,
            build_env,
        )
        return

    if emscripten_build:
        lock_file = root / "conan_emscripten.lock"
        logger.info("Emscripten build mode: cross-building via Conan")
        logger.info(f"Lock file: {lock_file}")
    else:
        lock_file = root / f"conan_{windowing}.lock"

    # Remove build configuration directory if -x flag is provided
    if args.get("rebuild") and build_dir.exists():
        logger.info(f"Rebuild flag (-x) detected. Removing build directory: {build_dir}")
        try:
            remove_tree_with_retries(build_dir)
        except PermissionError:
            logger.warning(
                f"Could not fully remove {build_dir} (files locked by another process). "
                "Continuing; dependency rebuild will be driven by host packages from the lock file."
            )

    # Create build directory if missing
    build_folder.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)

    preset_name = _cmake_preset_name(build_type, emscripten_build)

    # Isolate inherited Python overrides for Conan/CMake subprocesses.
    build_env = sanitized_subprocess_env()

    if args.get("build_only"):
        logger.info("Build only mode (-b): Skipping configuration steps")
        logger.info(f"Building with configuration: {build_type}")

        conanbuild = build_dir / "conanbuild"
        cmake_exe = find_executable("cmake")

        with CommandGroup("CMake build", env=build_env) as g:
            build_log_file = logs_dir / "cmake_build.log"
            build_args = [cmake_exe, "--build", "--preset", preset_name]
            g.run(build_args, log_file=build_log_file, env_script=conanbuild, cwd=root)
    else:
        ensure_conan_profile()
        export_local_conan_recipes(root, logs_dir, conan_config)

        if args.get("configure_only"):
            logger.info(f"Configuring with configuration: {build_type}")
        else:
            logger.info(f"Building with configuration: {build_type}")

        # Handle lock file generation and usage
        should_create_lock = args.get("update_lock") or not lock_file.exists()

        # Emscripten flags override the default host profile settings/conf
        emscripten_flags = (
            get_emscripten_conan_flags(root, build_folder) if emscripten_build else []
        )

        conan_exe = find_executable("conan")
        with CommandGroup("Conan dependencies", cwd=build_folder, env=build_env) as g:
            local_recipe_names = get_local_recipe_names(root, conan_config)
            if should_create_lock:
                if args.get("update_lock"):
                    logger.info("Update lock flag (-u) detected. Regenerating lock file...")
                else:
                    logger.info("Lock file not found. Generating new lock file...")
                lock_log_file = logs_dir / f"conan_lock_create_{windowing}.log"
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
                g.run(lock_args, log_file=lock_log_file)
                # Strip revisions for local recipes in the lock file
                strip_local_recipe_revisions(lock_file, local_recipe_names)
            else:
                logger.info(f"Lock file found. Using existing lock file: {lock_file}")

            install_log_file = logs_dir / "conan_install.log"

            # Determine --build flags: rebuild all host packages (-x) or only missing
            if args.get("rebuild"):
                host_pkgs = _host_package_names(lock_file)
                build_flags = [f"--build={name}/*" for name in host_pkgs]
                logger.info(f"Forcing rebuild of {len(host_pkgs)} host packages")
            else:
                build_flags = ["--build=missing"]
                # OpenEXR: Conan Center's prebuilt binary (Iex-3_3.lib) calls
                # `std::_Search_vectorized` from the MSVC STL it was built against;
                # linking against a locally-installed MSVC that inlines that helper
                # differently fails with an unresolved external. Always build from
                # source on Windows so STL internals match cl.exe.
                if sys.platform == "win32" and "--build=openexr/*" not in build_flags:
                    build_flags.append("--build=openexr/*")

            logger.info("Installing dependencies with Conan...")

            # Skip deployers when the lock file hasn't changed since the
            # last successful deploy -- avoids the full_deploy delete-and-
            # recopy that fails when another process holds a file handle.
            skip_deploy = _deploy_is_current(lock_file, conan_deps_root, build_type)
            deployer_flags: list[str] = []
            if skip_deploy:
                logger.info("Deploy is current (lock file unchanged) -- skipping deployers")
            else:
                deployer_flags = [
                    f"--deployer-folder={conan_deps_root}",
                    "--deployer=full_deploy",
                    "--deployer=runtime_deploy",
                ]

            conan_install_args = [
                conan_exe,
                "install",
                str(root),
                "--lockfile",
                str(lock_file),
                f"--output-folder={build_dir}",
                *deployer_flags,
                *build_flags,
                f"--profile:host={conan_profile}",
                f"--profile:build={conan_profile}",
                "-o",
                f"&:windowing={windowing}",
                "-s",
                "compiler.cppstd=17",
                "-s",
                f"build_type={build_type}",
                *emscripten_flags,
            ]
            g.run(conan_install_args, log_file=install_log_file)

            if not skip_deploy:
                _write_deploy_sentinel(lock_file, conan_deps_root, build_type)

        # Conan build environment script (vcvars on Windows for Ninja + MSVC)
        conanbuild = build_dir / "conanbuild"

        # Phase 1: configure + build host tools needed by prebuild steps
        host_tools = list(_HOST_TOOL_TARGETS)
        if host_tools and not emscripten_build:
            with CommandGroup("Host tools", cwd=build_folder, env=build_env) as g:
                cmake_exe = find_executable("cmake")
                ensure_cmake_file_api_query(build_folder / build_type)
                configure_args = [
                    cmake_exe,
                    "--preset",
                    preset_name,
                    "-S",
                    str(root),
                ]
                if usd_modules:
                    configure_args.append(f"-DPTS_USD_MODULES={';'.join(usd_modules)}")
                g.run(
                    configure_args,
                    log_file=logs_dir / "cmake_configure_tools.log",
                    env_script=conanbuild,
                )
                for tool_name in host_tools:
                    target = _HOST_TOOL_TARGETS[tool_name]["target"]
                    g.run(
                        [
                            cmake_exe,
                            "--build",
                            "--preset",
                            preset_name,
                            "--target",
                            target,
                        ],
                        log_file=logs_dir / f"cmake_build_{target}.log",
                        env_script=conanbuild,
                        cwd=root,
                    )

            # Mirror the --host-tools-only layout: stage the Conan env
            # scripts next to the built host-tool binaries so downstream
            # prebuild steps (e.g. slangc) can locate conanrun via a
            # single convention regardless of which build path built the
            # tool.
            bin_dir = build_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for script in build_dir.glob("conan*"):
                if script.is_file():
                    shutil.copy2(script, bin_dir / script.name)

        with CommandGroup("Generate assets"):
            prepare_assets(ctx)

        with CommandGroup("CMake configure", cwd=build_folder, env=build_env) as g:
            configure_log_file = logs_dir / "cmake_configure.log"
            cmake_exe = find_executable("cmake")
            ensure_cmake_file_api_query(build_folder / build_type)

            cmake_args = [
                cmake_exe,
                "--preset",
                preset_name,
                "-S",
                str(root),
            ]
            if emscripten_build:
                emdawnwebgpu_port = ensure_emdawnwebgpu_port(root, build_folder)
                cmake_args.append(f"-DEMDAWNWEBGPU_PORT_FILE={emdawnwebgpu_port}")
            if usd_modules:
                cmake_args.append(f"-DPTS_USD_MODULES={';'.join(usd_modules)}")

            g.run(cmake_args, log_file=configure_log_file, env_script=conanbuild)

        generate_cpp_properties(root, build_dir, windowing)

        if not args.get("configure_only"):
            with CommandGroup("CMake build", env=build_env) as g:
                build_log_file = logs_dir / "cmake_build.log"

                build_args = [cmake_exe, "--build", "--preset", preset_name]
                # Build presets require CMakeUserPresets.json at project root
                g.run(build_args, log_file=build_log_file, env_script=conanbuild, cwd=root)

        else:
            logger.info("Configure only mode (-c): Skipping build step")

    tests = discover_test_targets(build_dir)
    env_vars = load_conan_env(build_dir, preset_type="test")
    generate_launch_json(root, build_dir, build_type, tests, env_vars)
