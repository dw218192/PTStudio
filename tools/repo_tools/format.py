"""Format subcommand implementation."""

import argparse
import subprocess
import sys
from pathlib import Path

from repo_tools import (
    RepoContext,
    RepoTool,
    build_repo_context,
    find_venv_executable,
    load_repo_config,
    logger,
)


TARGET_EXTENSIONS = {".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".hxx"}
_ALWAYS_EXCLUDE = {"_tools", "ext", ".git", ".vs", "build"}


class FormatTool(RepoTool):
    name = "format"
    help = "Format source code using clang-format"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--verify",
            action="store_true",
            help=(
                "Verify that files are formatted correctly without modifying them "
                "(for CI/CD)"
            ),
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(platform=context["platform"], verify=False)

    def execute(self, args: argparse.Namespace) -> None:
        """Format subcommand implementation."""
        root = Path(__file__).parent.parent.parent
        config = load_repo_config(root)
        context = build_repo_context(root, "Debug", config, args.platform)

        exclude_dirs = set(_ALWAYS_EXCLUDE)
        exclude_dirs.add(Path(context["build_root"]).name)
        exclude_dirs.add(Path(context["logs_root"]).name)

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
                if not any(excluded in parts for excluded in exclude_dirs):
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
                original_content = file_path.read_text(
                    encoding="utf-8", errors="replace"
                )

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
                        capture_output=True,
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
