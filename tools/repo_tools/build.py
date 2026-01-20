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


def _collect_header_dirs(root: Path, base: Path) -> set[Path]:
    header_dirs: set[Path] = set()
    if not base.exists():
        return header_dirs
    for pattern in ("*.h", "*.hpp", "*.inl"):
        for header in base.rglob(pattern):
            header_dirs.add(header.parent)
    return header_dirs


def _generate_cpp_properties(
    root: Path, build_type: str, gapi: str, windowing: str
) -> None:
    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)

    include_dirs: set[Path] = set()
    include_dirs |= _collect_header_dirs(root, root / "core" / "include")
    include_dirs |= _collect_header_dirs(root, root / "core" / "api")
    include_dirs |= _collect_header_dirs(root, root / "editor" / "include")
    include_dirs |= _collect_header_dirs(root, root / "plugins")
    include_dirs |= _collect_header_dirs(root, root / "gl_utils")
    include_dirs |= _collect_header_dirs(root, root / "vulkan_raytracer")

    include_dirs |= _collect_header_dirs(
        root, root / "core" / "src" / "rendering" / gapi
    )
    include_dirs |= _collect_header_dirs(
        root, root / "core" / "src" / "rendering" / windowing
    )

    deps_include = root / "_build" / "deps" / "full_deploy" / "host"
    if deps_include.exists():
        include_dirs.add(deps_include)
        include_dirs |= _collect_header_dirs(root, deps_include)

    for cmrc_include in root.glob("_build/*/_cmrc/include"):
        include_dirs.add(cmrc_include)

    include_paths = sorted(
        {_format_workspace_path(root, path) for path in include_dirs if path.exists()}
    )

    defines = [
        "SPDLOG_FMT_EXTERNAL",
        f"PTS_GAPI_{gapi}",
        f"PTS_WINDOWING_{windowing}",
        f'PTS_GAPI="{gapi}"',
        f'PTS_WINDOWING="{windowing}"',
    ]
    if gapi == "vulkan":
        defines.append("VULKAN_HPP_DISPATCH_LOADER_DYNAMIC")

    config = {
        "name": "PTStudio",
        "cppStandard": "c++17",
        "cStandard": "c17",
        "defines": defines,
        "includePath": include_paths,
        "browse": {"path": include_paths},
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
    gapi = args.gapi
    windowing = args.windowing
    lock_file = root / f"conan_{gapi}_{windowing}.lock"

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
                lock_log_file = logs_dir / f"conan_lock_create_{gapi}_{windowing}.log"
                conan_exe = find_venv_executable("conan")
                run_command(
                    [
                        conan_exe,
                        "lock",
                        "create",
                        "..",
                        "-o",
                        f"&:gapi={gapi}",
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
                    f"&:gapi={gapi}",
                    "-o",
                    f"&:windowing={windowing}",
                    "-s",
                    "compiler.cppstd=17",
                    "-s",
                    f"build_type={args.build_type}",
                ],
                log_file=install_log_file,
            )

            _generate_cpp_properties(root, args.build_type, gapi, windowing)

            configure_log_file = logs_dir / "cmake_configure.log"
            cmake_exe = find_venv_executable("cmake")
            run_command(
                [
                    cmake_exe,
                    "-S",
                    "..",
                    "-B",
                    args.build_type,
                    "-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake",
                    f"-DCMAKE_BUILD_TYPE={args.build_type}",
                    "-DCMAKE_CXX_STANDARD=17",
                    "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                ],
                log_file=configure_log_file,
            )

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
    parser.add_argument(
        "--gapi",
        choices=["vulkan", "null"],
        default="vulkan",
        help="Graphics backend (default: vulkan)",
    )
    parser.add_argument(
        "--windowing",
        choices=["glfw", "null"],
        default="glfw",
        help="Windowing backend (default: glfw)",
    )
    parser.set_defaults(func=build_command)
