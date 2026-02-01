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
    RepoContext,
    RepoTool,
    apply_repo_tool_args,
    build_repo_context,
    create_repo_tool_args,
    ensure_conan_profile,
    find_venv_executable,
    get_repo_tool,
    get_repo_tool_config_args,
    is_windows,
    load_repo_config,
    normalize_env_config,
    normalize_repo_tool_args,
    print_tool,
    resolve_env_vars,
    run_command,
    logger,
)


def execute_build_steps(
    root: Path,
    config: dict,
    context: RepoContext,
    logs_dir: Path,
    steps_config: dict,
    step_type: str,
    current_tool: str,
) -> None:
    """Execute prebuild or postbuild steps defined in config.

    Args:
        root: Repository root path
        config: Full configuration dictionary
        context: Repository context
        logs_dir: Logs directory path
        steps_config: Dictionary of build steps from config
        step_type: Either "prebuild" or "postbuild" for logging
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

        tool = get_repo_tool(repo_tool)
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

        mock_args = create_repo_tool_args(repo_tool, context)

        tool_config_args = get_repo_tool_config_args(config, repo_tool)
        apply_repo_tool_args(mock_args, tool_config_args)

        step_args = normalize_repo_tool_args(step_args_value)
        apply_repo_tool_args(mock_args, step_args)

        if not hasattr(mock_args, "passthrough_args"):
            mock_args.passthrough_args = []

        try:
            tool.execute(mock_args)
            logger.info(f"  ✓ {step_name} completed")
        except Exception as e:
            logger.error(f"  ✗ {step_name} failed: {e}")
            raise RuntimeError(f"{step_type} step '{step_name}' failed") from e


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


def _find_package_in_deploy(conan_deps_root: Path, package_name: str) -> Path | None:
    """Find a package in the full_deploy folder, returning the deepest content directory."""
    host_dir = conan_deps_root / "full_deploy" / "host" / package_name
    if not host_dir.exists():
        return None
    # Navigate through version/build_type/arch structure to find the actual content
    # Structure: full_deploy/host/<package>/<version>/<build_type>/<arch>/
    current = host_dir
    while current.is_dir():
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        # Check if this looks like the package root (has lib/, bin/, include/, etc.)
        if any((current / d).exists() for d in ["lib", "bin", "include", "cmake"]):
            return current
        if len(subdirs) == 1:
            current = subdirs[0]
        elif len(subdirs) > 1:
            # Multiple subdirs - pick the first one (shouldn't happen normally)
            current = subdirs[0]
        else:
            break
    return None


def _ensure_package_installed(
    conan_deps_root: Path,
    package_name: str,
    install_dir: Path,
    marker_subpath: str,
    exclude_dirs: set[str] | None = None,
) -> bool:
    """Copy a package from full_deploy to the expected install location.

    Returns True if the package is installed (either already existed or was copied).
    """
    marker_path = install_dir / marker_subpath
    if marker_path.exists():
        print_tool(f"{package_name} install found: {install_dir}")
        return True

    deploy_dir = _find_package_in_deploy(conan_deps_root, package_name)
    if deploy_dir is None:
        print_tool(f"{package_name} not found in full_deploy folder")
        return False

    print_tool(f"Copying {package_name} from {deploy_dir} to {install_dir}")
    install_dir.mkdir(parents=True, exist_ok=True)

    def ignore_dirs(directory: str, names: list[str]) -> list[str]:
        if not exclude_dirs:
            return []
        return [name for name in names if name in exclude_dirs]

    ignore = ignore_dirs if exclude_dirs else None
    shutil.copytree(deploy_dir, install_dir, dirs_exist_ok=True, ignore=ignore)
    return True


def _export_local_conan_recipes(
    root: Path, logs_dir: Path, conan_config: dict
) -> None:
    recipes = conan_config.get("local_recipes", [])
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




def _get_local_recipe_names(conan_config: dict) -> set[str]:
    recipes = conan_config.get("local_recipes", [])
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
    import json

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
        print_tool(
            "Stripped local recipe revisions in lock file: "
            f"{', '.join(sorted(local_recipe_names))}"
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


def _get_dict_arg(args: argparse.Namespace, field_name: str) -> dict:
    """Extract a dict argument from args, warn if non-dict, return {} if None or invalid."""
    value = getattr(args, field_name, None)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    logger.warning(f"Build arg '{field_name}' must be a dict; ignoring.")
    return {}


def build_command(args: argparse.Namespace, current_tool: str) -> None:
    """
    Meta-meta-build system implementation.
    1. Configure the project
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
    windowing = args.windowing
    lock_file = Path(context["conan_lock"])
    conan_deps_root = Path(context["conan_deps_root"])

    conan_config = _get_dict_arg(args, "conan")
    prebuild_steps = _get_dict_arg(args, "prebuild")
    postbuild_steps = _get_dict_arg(args, "postbuild")

    # Remove build configuration directory if -x flag is provided
    if args.rebuild and build_dir.exists():
        print_tool(f"Rebuild flag (-x) detected. Removing build directory: {build_dir}")
        _remove_tree_with_retries(build_dir)

    # Create build directory if missing
    build_folder.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Change to build directory
    original_cwd = os.getcwd()
    try:
        os.chdir(build_folder)

        if args.build_only:
            print_tool("Build only mode (-b): Skipping configuration steps")
            print_tool(f"Building with configuration: {args.build_type}")
        else:
            ensure_conan_profile()
            _export_local_conan_recipes(root, logs_dir, conan_config)

            if args.configure_only:
                print_tool(f"Configuring with configuration: {args.build_type}")
            else:
                print_tool(f"Building with configuration: {args.build_type}")

            # Handle lock file generation and usage
            should_create_lock = args.update_lock or not lock_file.exists()

            local_recipe_names = _get_local_recipe_names(conan_config)
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
                # Strip revisions for local recipes in the lock file
                _strip_local_recipe_revisions(lock_file, local_recipe_names)
            else:
                print_tool(f"Lock file found. Using existing lock file: {lock_file}")

            install_log_file = logs_dir / "conan_install.log"
            conan_exe = find_venv_executable("conan")

            print_tool(f"Installing dependencies with Conan...")
            run_command(
                [
                    conan_exe,
                    "install",
                    "..",
                    "--lockfile",
                    str(lock_file),
                    f"--output-folder={args.build_type}",
                    f"--deployer-folder={conan_deps_root}",
                    "--deployer=full_deploy",
                    "--build=missing",
                    f"--profile:host={args.conan_profile}",
                    f"--profile:build={args.conan_profile}",
                    "-o",
                    f"&:windowing={windowing}",
                    "-s",
                    "compiler.cppstd=17",
                    "-s",
                    f"build_type={args.build_type}",
                ],
                log_file=install_log_file,
            )

            # Ensure Dawn and OpenUSD are deployed
            dawn_deploy_dir = _find_package_in_deploy(conan_deps_root, "dawn")
            if not dawn_deploy_dir:
                raise RuntimeError("Failed to find Dawn package in full_deploy")

            openusd_deploy_dir = _find_package_in_deploy(conan_deps_root, "openusd")
            if not openusd_deploy_dir:
                raise RuntimeError("Failed to find OpenUSD package in full_deploy")

            # Execute prebuild steps
            if prebuild_steps:
                print_tool("Running prebuild steps...")
                execute_build_steps(
                    root,
                    config,
                    context,
                    logs_dir,
                    prebuild_steps,
                    "prebuild",
                    current_tool,
                )

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
                "-DCMAKE_CXX_STANDARD=17",
                "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
            # Find Dawn cmake directory from deployed package
            dawn_cmake_dir = dawn_deploy_dir / "lib" / "cmake" / "Dawn"
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

            # Execute postbuild steps
            if postbuild_steps:
                print_tool("Running postbuild steps...")
                execute_build_steps(
                    root,
                    config,
                    context,
                    logs_dir,
                    postbuild_steps,
                    "postbuild",
                    current_tool,
                )
        else:
            print_tool("Configure only mode (-c): Skipping build step")

        tests = _discover_test_targets(build_dir)
        launch_args = get_repo_tool_config_args(config, "launch")
        env_config = normalize_env_config(launch_args.get("env"))
        env_vars = resolve_env_vars(env_config, context)
        _generate_launch_json(root, build_dir, args.build_type, tests, env_vars)
    finally:
        os.chdir(original_cwd)


class BuildTool(RepoTool):
    name = "build"
    help = "Build the project"

    def setup(self, parser: argparse.ArgumentParser) -> None:
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
            help=(
                "Configure only flag: runs conan install and cmake configure, "
                "but skips building"
            ),
        )
        parser.add_argument(
            "-b",
            "--build-only",
            action="store_true",
            help=(
                "Build only flag: skips conan install and cmake configure, "
                "only runs build"
            ),
        )
        parser.add_argument(
            "--build-type",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration type (default: Debug)",
        )
        parser.add_argument(
            "--conan-profile",
            help="Conan profile (default: default)",
        )
        parser.add_argument(
            "--windowing",
            choices=["glfw", "null"],
            help="Windowing backend (default: glfw)",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(
            rebuild=False,
            update_lock=False,
            configure_only=False,
            build_only=False,
            build_type=context["build_type"],
            conan_profile="default",
            windowing="glfw",
            prebuild={},
            postbuild={},
            conan={},
        )

    def execute(self, args: argparse.Namespace) -> None:
        build_command(args, self.name)
