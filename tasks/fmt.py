"""Format source code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from tasks.utils import ProjectContext, find_executable, logger
from tasks.utils.project import load_context, project_options

_CLANG_FORMAT_EXTENSIONS = {".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".hxx"}
_BATCH_SIZE = 200  # max files per clang-format invocation (Windows cmdline limit)


def _git_tracked_files(root: Path, extensions: set[str]) -> list[Path] | None:
    """Return tracked files matching *extensions*, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    files = []
    for entry in result.stdout.split("\0"):
        if not entry:
            continue
        p = root / entry
        if p.suffix in extensions:
            files.append(p)
    return files


def run(ctx: ProjectContext, args: dict[str, Any]) -> None:
    args = ctx.arguments("format", {"verify": False}, args)
    root = ctx.workspace_root
    verify = args.get("verify", False)
    _run_clang_format(root, verify)
    _format_python_tasks(root, verify)


def _collect_files(root: Path, extensions: set[str]) -> list[Path]:
    """Collect source files, respecting .gitignore when in a git repo."""
    files = _git_tracked_files(root, extensions)
    if files is not None:
        return files
    # Fallback for non-git repos: rglob with minimal exclusion
    result = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in extensions:
            parts = path.parts
            if not any(p.startswith(".") for p in parts):
                result.append(path)
    return result


def _run_clang_format(
    root: Path,
    verify: bool,
    extensions: set[str] | None = None,
) -> None:
    if extensions is None:
        extensions = _CLANG_FORMAT_EXTENSIONS

    clang_format_exe = find_executable("clang-format")

    clang_format_file = root / ".clang-format"
    if not clang_format_file.exists():
        logger.error(f".clang-format not found at {clang_format_file}")
        sys.exit(1)

    source_files = _collect_files(root, extensions)

    if not source_files:
        logger.warning("No source files found to format")
        return

    logger.info(f"Found {len(source_files)} source files to format")

    if verify:
        _clang_format_verify(clang_format_exe, clang_format_file, source_files)
    else:
        _clang_format_inplace(clang_format_exe, clang_format_file, source_files)


def _clang_format_verify(
    exe: str,
    style_file: Path,
    files: list[Path],
) -> None:
    logger.info("Running in verify mode (no files will be modified)")
    failed_files = []

    # Try --dry-run --Werror first (clang-format 10+)
    test = subprocess.run(
        [exe, "--dry-run", "--Werror", "--style=file", str(files[0])],
        capture_output=True,
        text=True,
    )
    use_dry_run = test.returncode in (0, 1)  # 0=ok, 1=diff found; not "unknown flag"

    if use_dry_run:
        for batch_start in range(0, len(files), _BATCH_SIZE):
            batch = files[batch_start : batch_start + _BATCH_SIZE]
            result = subprocess.run(
                [exe, "--dry-run", "--Werror", f"--style=file:{style_file}"]
                + [str(f) for f in batch],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                logger.error(result.stderr or "clang-format failed")
                failed_files.extend(batch)
                # Parse stderr for file names
                for line in result.stderr.splitlines():
                    for f in batch:
                        if str(f) in line:
                            failed_files.append(f)
                            break
    else:
        # Fallback: per-file comparison
        for file_path in files:
            original_content = file_path.read_text(encoding="utf-8", errors="replace")
            result = subprocess.run(
                [exe, f"--style=file:{style_file}", str(file_path)],
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
            if original_content != result.stdout:
                failed_files.append(file_path)
                logger.error(f"File is not properly formatted: {file_path}")

    if failed_files:
        logger.error(f"{len(failed_files)} file(s) are not properly formatted")
        sys.exit(1)
    else:
        logger.info("All files are properly formatted")


def _clang_format_inplace(
    exe: str,
    style_file: Path,
    files: list[Path],
) -> None:
    logger.info("Formatting files...")
    for batch_start in range(0, len(files), _BATCH_SIZE):
        batch = files[batch_start : batch_start + _BATCH_SIZE]
        try:
            subprocess.run(
                [exe, "-i", f"--style=file:{style_file}"] + [str(f) for f in batch],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"Failed to format batch: {error_msg}")
            sys.exit(1)
    logger.info(f"Successfully formatted {len(files)} file(s)")


def _format_python_tasks(root: Path, verify: bool) -> None:
    executable = find_executable("ruff")
    tasks_dir = str(root / "tasks")
    check = [executable, "check", tasks_dir]
    format_args = [executable, "format", tasks_dir]
    if verify:
        format_args.append("--check")
    else:
        check.append("--fix")
    for command in (check, format_args):
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)


@click.command(name="fmt", help="Format source code")
@project_options
@click.option("--verify", is_flag=True, help="Check formatting without modifying files")
@click.pass_context
def main(cli: click.Context, platform: str, build_type: str, **args: Any) -> None:
    context = load_context(platform, args.get("config") or build_type, passthrough=cli.args)
    run(context, args)


if __name__ == "__main__":
    main()
