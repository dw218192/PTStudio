"""Build subcommand implementation."""
import argparse
import os
import shutil
from pathlib import Path

from repo_tools import ensure_conan_profile, find_venv_executable, run_command


def build_command(args: argparse.Namespace) -> None:
    """Build subcommand implementation."""
    root = Path(__file__).parent.parent.parent
    build_folder = root / "_build"
    logs_dir = root / "_logs"
    lock_file = root / "conan.lock"

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
                lock_log_file = logs_dir / "conan_lock_create.log"
                conan_exe = find_venv_executable("conan")
                run_command(
                    [
                        conan_exe,
                        "lock",
                        "create",
                        "..",
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
                    "-s",
                    "compiler.cppstd=17",
                    "-s",
                    f"build_type={args.build_type}",
                ],
                log_file=install_log_file,
            )

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
    parser.set_defaults(func=build_command)
