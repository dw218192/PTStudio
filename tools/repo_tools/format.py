"""Format subcommand implementation."""

import argparse
import subprocess
import sys
from pathlib import Path

from repo_tools import find_venv_executable, logger


TARGET_EXTENSIONS = {".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".hxx"}
EXCLUDE_DIRS = {"_build", "_tools", "_logs", "ext", ".git", ".vs", "build"}


def format_command(args: argparse.Namespace) -> None:
    """Format subcommand implementation."""
    root = Path(__file__).parent.parent.parent
    clang_format_exe = find_venv_executable("clang-format")
    clang_format_file = root / ".clang-format"

    if not clang_format_file.exists():
        logger.error(f".clang-format not found at {clang_format_file}")
        sys.exit(1)

    # Find all C/C++ source files
    source_files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in TARGET_EXTENSIONS:
            # Check if path is in any excluded directory
            parts = path.parts
            if not any(excluded in parts for excluded in EXCLUDE_DIRS):
                source_files.append(path)

    if not source_files:
        logger.warning("No C/C++ source files found to format")
        return

    logger.info(f"Found {len(source_files)} source files to format")

    if args.verify:
        # Verify mode: check if files are formatted correctly
        logger.info("Running in verify mode (no files will be modified)")
        failed_files = []
        for file_path in source_files:
            # Read original file content
            original_content = file_path.read_text(encoding="utf-8", errors="replace")

            # Get formatted content
            result = subprocess.run(
                [
                    clang_format_exe,
                    f"--style=file:{clang_format_file}",
                    str(file_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:
                failed_files.append(file_path)
                logger.error(f"Failed to format {file_path}: {result.stderr}")
                continue

            formatted_content = result.stdout
            if original_content != formatted_content:
                failed_files.append(file_path)
                logger.error(f"File is not properly formatted: {file_path}")

        if failed_files:
            logger.error(f"{len(failed_files)} file(s) are not properly formatted")
            sys.exit(1)
        else:
            logger.info("All files are properly formatted")
    else:
        # Format mode: actually format the files
        logger.info("Formatting files...")
        for file_path in source_files:
            try:
                subprocess.run(
                    [
                        clang_format_exe,
                        "-i",
                        f"--style=file:{clang_format_file}",
                        str(file_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                logger.debug(f"Formatted: {file_path}")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else str(e)
                logger.error(f"Failed to format {file_path}: {error_msg}")
                sys.exit(1)
        logger.info(f"Successfully formatted {len(source_files)} file(s)")


def register_format_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the format subcommand."""
    parser = subparsers.add_parser(
        "format", help="Format source code using clang-format"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that files are formatted correctly without modifying them (for CI/CD)",
    )
    parser.set_defaults(func=format_command)
