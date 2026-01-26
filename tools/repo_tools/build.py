"""Build subcommand implementation."""

import argparse
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

from repo_tools import (
    build_repo_context,
    ensure_conan_profile,
    find_venv_executable,
    is_windows,
    load_repo_config,
    print_tool,
    resolve_env_vars,
    resolve_path,
    resolve_slang_test_output,
    run_command,
    resolve_slang_shaders,
    find_slangc,
    logger,
    RepoContext,
)


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


def _map_usd_build_variant(build_type: str) -> str:
    build_type = build_type.lower()
    if build_type == "debug":
        return "debug"
    if build_type == "relwithdebinfo":
        return "relwithdebuginfo"
    return "release"


def _get_vcvars_path() -> Path | None:
    if not is_windows():
        return None

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.exists():
        return None

    try:
        install_path = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    if not install_path:
        return None

    vcvars = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if vcvars.exists():
        return vcvars
    return None


def _remove_tree_with_retries(
    path: Path, attempts: int = 5, delay: float = 1.0
) -> None:
    if not path.exists():
        return

    def remove_file(target: Path) -> None:
        try:
            target.unlink()
        except PermissionError:
            os.chmod(target, stat.S_IWRITE)
            target.unlink()

    def remove_dir(target: Path) -> None:
        try:
            target.rmdir()
        except PermissionError:
            os.chmod(target, stat.S_IWRITE)
            target.rmdir()

    def remove_tree(target: Path) -> None:
        for root_dir, dirnames, filenames in os.walk(target, topdown=False):
            for filename in filenames:
                remove_file(Path(root_dir) / filename)
            for dirname in dirnames:
                remove_dir(Path(root_dir) / dirname)
        remove_dir(target)

    for attempt in range(1, attempts + 1):
        try:
            remove_tree(path)
            return
        except PermissionError as error:
            offending = error.filename or str(path)
            print_tool(f"Remove failed for: {offending}")
            if attempt == attempts:
                raise
            print_tool(
                f"Remove failed (attempt {attempt}/{attempts}) due to locked files. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


def _ensure_openusd(
    root: Path,
    build_type: str,
    logs_dir: Path,
    install_dir: Path,
    usd_build_dir: Path,
    force_rebuild: bool = False,
) -> Path:
    usd_root = root / "ext" / "usd"
    build_script = usd_root / "build_scripts" / "build_usd.py"
    if not build_script.exists():
        raise FileNotFoundError(f"OpenUSD build script not found: {build_script}")

    pxr_config = install_dir / "pxrConfig.cmake"
    if pxr_config.exists():
        if not force_rebuild:
            print_tool(f"OpenUSD install found: {install_dir}")
            return install_dir
        print_tool(f"OpenUSD install found; forcing rebuild: {install_dir}")
        usd_root_dir = usd_build_dir.parent
        if usd_root_dir.exists():
            _remove_tree_with_retries(usd_root_dir)

    print_tool("OpenUSD install not found. Building OpenUSD...")
    usd_log_file = logs_dir / f"openusd_build_{build_type.lower()}.log"
    python_exe = find_venv_executable("python")
    usd_cmd = [
        python_exe,
        str(build_script),
        str(install_dir),
        "--build",
        str(usd_build_dir),
        "--build-variant",
        _map_usd_build_variant(build_type),
        "--no-python",
        "--no-python-docs",
        "--no-docs",
        "--no-tests",
        "--no-examples",
        "--no-tutorials",
        "--no-tools",
        "--no-imaging",
        "--no-usdview",
        "--no-ptex",
        "--no-openvdb",
        "--no-openimageio",
        "--no-opencolorio",
        "--no-alembic",
        "--no-materialx",
        "--no-embree",
        "--no-prman",
        "--no-vulkan",
    ]

    if is_windows() and shutil.which("cl") is None:
        vcvars = _get_vcvars_path()
        if vcvars is None:
            raise RuntimeError(
                "MSVC compiler not found. Run from a Developer Command Prompt "
                "or install Visual Studio with C++ tools."
            )
        cmd_line = subprocess.list2cmdline(usd_cmd)
        usd_cmd_script = logs_dir / f"openusd_build_{build_type.lower()}.cmd"
        usd_cmd_script.write_text(
            f'@echo off\ncall "{vcvars}"\n{cmd_line}\nexit /b %errorlevel%\n',
            encoding="utf-8",
        )
        run_command(
            ["cmd.exe", "/c", str(usd_cmd_script)],
            log_file=usd_log_file,
        )
    else:
        run_command(usd_cmd, log_file=usd_log_file)

    if not pxr_config.exists():
        raise RuntimeError(f"OpenUSD install failed: {pxr_config} not found.")

    return install_dir


def _export_local_conan_recipes(root: Path, logs_dir: Path, config: dict) -> None:
    recipes = config.get("conan", {}).get("local_recipes", [])
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


def _validate_slang_test_output(root: Path, config: dict, context: dict) -> None:
    output_path = resolve_slang_test_output(root, config, context)
    if output_path is None:
        return
    if not output_path.exists():
        raise FileNotFoundError(f"Slang test output not found: {output_path}")

    contents = output_path.read_text(encoding="utf-8", errors="replace").strip()
    if not contents:
        raise RuntimeError(f"Slang test output is empty: {output_path}")
    if "@vertex" not in contents:
        raise RuntimeError(
            f"Slang test output missing '@vertex' entry point: {output_path}"
        )


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


def compile_slang_shaders(
    root: Path, config: dict, context: RepoContext, logs_dir: Path
) -> None:
    shaders = resolve_slang_shaders(root, config, context)
    if not shaders:
        return

    slangc_path = find_slangc(root, config, context)
    if slangc_path is None:
        raise RuntimeError("slangc not found. Install via Conan or set compiler_path.")

    for input_path, output_path in shaders:
        if not input_path.exists():
            raise FileNotFoundError(f"Slang shader not found: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            if output_path.stat().st_mtime >= input_path.stat().st_mtime:
                continue
        log_file = logs_dir / f"slangc_{input_path.stem}.log"
        run_command(
            [
                str(slangc_path),
                str(input_path),
                "-target",
                "wgsl",
                "-o",
                str(output_path),
            ],
            log_file=log_file,
        )


def build_command(args: argparse.Namespace) -> None:
    """
    Meta-meta-build system implementation.
    1. Configure the project
      - build OpenUSD if needed (force rebuild if `--configure-only`)
      - fetch dependencies with conan
      - configure CMake
      - generate vscode cpp include paths for the project (compile_commands.json is not reliably generated for certain configurations)
    2. build the project using CMake by running the appropriate build system
    3. generate vscode launch configurations for the project
    """
    root = Path(__file__).parent.parent.parent
    config = load_repo_config(root)
    context = build_repo_context(root, args.build_type, config)
    build_folder = Path(context["build_root"])
    build_dir = Path(context["build_dir"])
    logs_dir = Path(context["logs_root"])
    windowing = config.get("build", {}).get("windowing", "glfw")
    lock_file = Path(context["conan_lock"])
    conan_deployer = config.get("build", {}).get("conan_deployer", "full_deploy")
    conan_deployer_folder = resolve_path(
        root,
        config.get("build", {}).get("conan_deployer_folder", "{conan_deps_root}"),
        context,
    )
    usd_install_dir = Path(context["usd_install_dir"])
    usd_build_dir = Path(context["usd_build_dir"])
    dawn_install_dir = Path(context["dawn_install_dir"])

    # Remove build configuration directory if -x flag is provided
    if args.rebuild and build_dir.exists():
        print_tool(f"Rebuild flag (-x) detected. Removing build directory: {build_dir}")
        _remove_tree_with_retries(build_dir)

    # Create build directory if missing
    build_folder.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)
    usd_install_dir = _ensure_openusd(
        root,
        args.build_type,
        logs_dir,
        usd_install_dir,
        usd_build_dir,
        force_rebuild=args.configure_only,
    )

    # Change to build directory
    original_cwd = os.getcwd()
    try:
        os.chdir(build_folder)

        if args.build_only:
            print_tool("Build only mode (-b): Skipping configuration steps")
            print_tool(f"Building with configuration: {args.build_type}")
        else:
            ensure_conan_profile()
            _export_local_conan_recipes(root, logs_dir, config)

            if args.configure_only:
                print_tool(f"Configuring with configuration: {args.build_type}")
            else:
                print_tool(f"Building with configuration: {args.build_type}")

            # Handle lock file generation and usage
            should_create_lock = args.update_lock or not lock_file.exists()

            if should_create_lock:
                if args.update_lock:
                    print_tool(
                        "Update lock flag (-u) detected. Regenerating lock file..."
                    )
                else:
                    print_tool("Lock file not found. Generating new lock file...")
                lock_log_file = logs_dir / f"conan_lock_create_{windowing}.log"
                conan_exe = find_venv_executable("conan")
                run_command(
                    [
                        conan_exe,
                        "lock",
                        "create",
                        "..",
                        "-o",
                        f"&:windowing={windowing}",
                        "--lockfile-out",
                        str(lock_file),
                    ],
                    log_file=lock_log_file,
                )
            else:
                print_tool(f"Lock file found. Using existing lock file: {lock_file}")

            install_log_file = logs_dir / "conan_install.log"
            conan_exe = find_venv_executable("conan")
            run_command(
                [
                    conan_exe,
                    "install",
                    "..",
                    "--lockfile",
                    str(lock_file),
                    f"--output-folder={args.build_type}",
                    f"--deployer-folder={conan_deployer_folder}",
                    f"--deployer={conan_deployer}",
                    "--build=missing",
                    f"--profile:host={args.conan_profile}",
                    f"--profile:build={args.conan_profile}",
                    "-o",
                    f"&:windowing={windowing}",
                    "-s",
                    "compiler.cppstd=17",
                    "-s",
                    f"build_type={args.build_type}",
                    "-c",
                    f"user.dawn:install_dir={dawn_install_dir.as_posix()}",
                ],
                log_file=install_log_file,
            )

            print_tool("Compiling slang shaders...")
            compile_slang_shaders(root, config, context, logs_dir)
            _validate_slang_test_output(root, config, context)

            print_tool("Building targets...")

            configure_log_file = logs_dir / "cmake_configure.log"
            cmake_exe = find_venv_executable("cmake")
            _ensure_cmake_file_api_query(build_folder / args.build_type)
            cmake_args = [
                cmake_exe,
                "-S",
                "..",
                "-B",
                args.build_type,
                "-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake",
                f"-DCMAKE_BUILD_TYPE={args.build_type}",
                f"-DPTS_WINDOWING={windowing}",
                f"-Dpxr_DIR={usd_install_dir}",
                "-DCMAKE_CXX_STANDARD=17",
                "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
            dawn_cmake_dir = dawn_install_dir / "lib" / "cmake" / "Dawn"
            if dawn_cmake_dir.exists():
                cmake_args.append(f"-DDawn_DIR={dawn_cmake_dir}")
            run_command(cmake_args, log_file=configure_log_file)

            _generate_cpp_properties(root, build_dir, windowing)

        if not args.configure_only:
            build_log_file = logs_dir / "cmake_build.log"
            cmake_exe = find_venv_executable("cmake")
            run_command(
                [
                    cmake_exe,
                    "--build",
                    args.build_type,
                    "--config",
                    args.build_type,
                ],
                log_file=build_log_file,
            )
        else:
            print_tool("Configure only mode (-c): Skipping build step")

        tests = _discover_test_targets(build_dir)
        env_vars = resolve_env_vars(config, context)
        _generate_launch_json(root, build_dir, args.build_type, tests, env_vars)
    finally:
        os.chdir(original_cwd)


def register_build_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the build subcommand."""
    parser = subparsers.add_parser("build", help="Build the project")
    parser.add_argument(
        "-x",
        "--rebuild",
        action="store_true",
        help="Rebuild flag: removes build configuration folder before building",
    )
    parser.add_argument(
        "-u",
        "--update-lock",
        action="store_true",
        help="Update lock flag: forces regeneration of conan.lock",
    )
    parser.add_argument(
        "-c",
        "--configure-only",
        action="store_true",
        help="Configure only flag: runs conan install and cmake configure, but skips building",
    )
    parser.add_argument(
        "-b",
        "--build-only",
        action="store_true",
        help="Build only flag: skips conan install and cmake configure, only runs build",
    )
    parser.add_argument(
        "--build-type",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        default="Debug",
        help="Build configuration type (default: Debug)",
    )
    parser.add_argument(
        "--conan-profile", default="default", help="Conan profile (default: default)"
    )
    parser.set_defaults(func=build_command)
