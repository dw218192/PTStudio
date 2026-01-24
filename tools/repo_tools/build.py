"""Build subcommand implementation."""

import argparse
import json
import os
import shutil
from pathlib import Path

from repo_tools import ensure_conan_profile, find_venv_executable, run_command


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
    target_names: set[str],
    include_plugins: bool,
) -> tuple[set[Path], set[str]]:
    include_dirs: set[Path] = set()
    defines: set[str] = set()
    plugins_root = root / "plugins"
    for config in codemodel.get("configurations", []):
        for target in config.get("targets", []):
            target_name = target.get("name")
            if not target_name:
                continue
            if target_name not in target_names:
                target_json = json.loads(
                    (reply_dir / target["jsonFile"]).read_text(encoding="utf-8")
                )
                if not include_plugins or not _target_has_plugin_sources(
                    target_json, plugins_root
                ):
                    continue
            else:
                target_json = json.loads(
                    (reply_dir / target["jsonFile"]).read_text(encoding="utf-8")
                )
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
            {"core", "editor"},
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


def build_command(args: argparse.Namespace) -> None:
    """Build subcommand implementation."""
    root = Path(__file__).parent.parent.parent
    build_folder = root / "_build"
    logs_dir = root / "_logs"
    windowing = "glfw"
    lock_file = root / "conan_glfw.lock"

    # Remove build directory if -x flag is provided
    if args.rebuild and build_folder.exists():
        print("Rebuild flag (-x) detected. Removing build folder...")
        shutil.rmtree(build_folder)

    # Create build directory if missing
    build_folder.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Change to build directory
    original_cwd = os.getcwd()
    try:
        os.chdir(build_folder)

        if args.build_only:
            print(f"Build only mode (-b): Skipping configuration steps")
            print(f"Building with configuration: {args.build_type}")
        else:
            ensure_conan_profile()

            if args.configure_only:
                print(f"Configuring with configuration: {args.build_type}")
            else:
                print(f"Building with configuration: {args.build_type}")

            # Handle lock file generation and usage
            should_create_lock = args.update_lock or not lock_file.exists()

            if should_create_lock:
                if args.update_lock:
                    print("Update lock flag (-u) detected. Regenerating lock file...")
                else:
                    print("Lock file not found. Generating new lock file...")
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
                print(f"Lock file found. Using existing lock file: {lock_file}")

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
                    "--deployer-folder=deps",
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

            configure_log_file = logs_dir / "cmake_configure.log"
            cmake_exe = find_venv_executable("cmake")
            _ensure_cmake_file_api_query(build_folder / args.build_type)
            run_command(
                [
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
                ],
                log_file=configure_log_file,
            )

            _generate_cpp_properties(root, build_folder / args.build_type, windowing)

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
            print("Configure only mode (-c): Skipping build step")
    finally:
        os.chdir(original_cwd)


def register_build_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the build subcommand."""
    parser = subparsers.add_parser("build", help="Build the project")
    parser.add_argument(
        "-x",
        "--rebuild",
        action="store_true",
        help="Rebuild flag: removes build folder before building",
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
