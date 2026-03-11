"""Build orchestrator — main build_command and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_tools.core import (
    CommandGroup,
    ToolContext,
    find_venv_executable,
    get_tool,
    invoke_tool,
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


# ── Prebuild / Postbuild Steps ───────────────────────────────────────


def execute_build_steps(
    _root: Path,
    config: dict,
    tokens: dict[str, str],
    dimensions: dict[str, str],
    _logs_dir: Path,
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


# ── Helpers ──────────────────────────────────────────────────────────


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


def _cmake_preset_name(build_type: str, emscripten: bool) -> str:
    bt = build_type.lower()
    return f"conan-emscripten-{bt}" if emscripten else f"conan-{bt}"


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
    build_root = Path(tokens["build_root"])
    build_folder = build_root / platform_id
    build_dir = Path(tokens["build_dir"])
    logs_dir = Path(tokens["logs_root"])
    windowing = args.get("windowing", "glfw")
    usd_modules = args.get("usd_modules") or []

    # Conan-specific paths from tokens
    conan_deps_root = Path(tokens["conan_deps_root"])

    conan_config = args.get("conan") or {}
    prebuild_steps = args.get("prebuild") or {}
    postbuild_steps = args.get("postbuild") or {}

    conan_profile = args.get("conan_profile", "default")

    # Emscripten build configuration
    emscripten_build = platform_id == "emscripten"
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

    # Prevent the repo-tool venv from contaminating Conan/CMake subprocesses
    build_env = sanitized_subprocess_env()

    if args.get("build_only"):
        logger.info("Build only mode (-b): Skipping configuration steps")
        logger.info(f"Building with configuration: {build_type}")

        conanbuild = build_dir / "conanbuild"
        cmake_exe = find_venv_executable("cmake")

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
        emscripten_flags = get_emscripten_conan_flags(root, build_folder) if emscripten_build else []

        conan_exe = find_venv_executable("conan")
        with CommandGroup("Conan dependencies", cwd=build_folder, env=build_env) as g:
            local_recipe_names = get_local_recipe_names(root, conan_config)
            if should_create_lock:
                if args.get("update_lock"):
                    logger.info(
                        "Update lock flag (-u) detected. Regenerating lock file..."
                    )
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

            logger.info("Installing dependencies with Conan...")
            conan_install_args = [
                conan_exe,
                "install",
                str(root),
                "--lockfile",
                str(lock_file),
                f"--output-folder={build_dir}",
                f"--deployer-folder={conan_deps_root}",
                "--deployer=full_deploy",
                "--deployer=runtime_deploy",
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

        # Execute prebuild steps
        if prebuild_steps:
            with CommandGroup("Prebuild steps"):
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

        with CommandGroup("CMake configure", cwd=build_folder, env=build_env) as g:
            configure_log_file = logs_dir / "cmake_configure.log"
            cmake_exe = find_venv_executable("cmake")
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

            # Execute postbuild steps
            if postbuild_steps:
                with CommandGroup("Postbuild steps"):
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

    tests = discover_test_targets(build_dir)
    env_vars = load_conan_env(build_dir, preset_type="test")
    generate_launch_json(root, build_dir, build_type, tests, env_vars)
