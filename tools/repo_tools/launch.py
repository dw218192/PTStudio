"""Launch subcommand implementation - runs executables."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from repo_tools.core import (
    RepoTool,
    ToolContext,
    detect_platform_identifier,
    is_windows,
    log_section,
    logger,
    normalize_build_type,
)


def _interactive_select(exe_paths: list[Path]) -> Path | None:
    """Show interactive menu to select an executable."""
    from InquirerPy import inquirer

    choices = [exe.stem for exe in sorted(exe_paths)]
    if not choices:
        return None

    try:
        selected = inquirer.select(
            message="Select executable to launch:",
            choices=choices,
        ).execute()
    except KeyboardInterrupt:
        return None

    if selected is None:
        return None

    for exe in exe_paths:
        if exe.stem == selected:
            return exe
    return None


def _discover_executables(target_dir: Path, is_emscripten: bool = False) -> list[Path]:
    """Discover executable files in a directory recursively."""
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    exe_paths = []
    try:
        for file in target_dir.rglob("*"):
            if not file.is_file():
                continue
            if is_emscripten:
                if file.suffix.lower() in (".html", ".js"):
                    exe_paths.append(file)
            elif is_windows():
                if file.suffix.lower() == ".exe":
                    exe_paths.append(file)
            else:
                try:
                    if os.access(file, os.X_OK):
                        exe_paths.append(file)
                except OSError:
                    continue
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not scan {target_dir}: {e}")

    if is_emscripten:
        # Each Emscripten target emits .html + .js; deduplicate by stem, keep .js
        by_stem: dict[str, Path] = {}
        for f in exe_paths:
            if f.stem not in by_stem or f.suffix.lower() == ".js":
                by_stem[f.stem] = f
        exe_paths = list(by_stem.values())

    return exe_paths


def _resolve_runtime_deploy_dir(build_dir: Path) -> Path | None:
    """Return the runtime_deploy directory if it exists.

    When Conan's ``runtime_deploy`` deployer is used, shared libraries are
    copied into the deployer-folder (typically ``deps/``) as a flat directory.
    This can be added to PATH directly, without needing a conanrun script.
    """
    deps_dir = build_dir.parent / "deps"
    if not deps_dir.exists():
        return None
    # Check for at least one shared library to confirm runtime_deploy was used
    dll_ext = ".dll" if is_windows() else ".so"
    for _ in deps_dir.glob(f"*{dll_ext}"):
        return deps_dir
    return None


def _resolve_env_script(build_dir: Path, is_emscripten: bool) -> Path | None:
    """Return the correct Conan env script for the platform.

    Native builds use ``conanrun`` (runtime DLL paths).
    Emscripten builds use ``conanbuild`` (emsdk tools: node, emrun).
    """
    name = "conanbuild" if is_emscripten else "conanrun"
    script = build_dir / name
    suffix = ".bat" if is_windows() else ".sh"
    resolved = script.with_suffix(suffix)
    return resolved if resolved.exists() else None


def _shell_wrap(cmd: list[str], env_script: Path | None) -> tuple[list[str] | str, bool]:
    """Wrap a command to source an env script if available.

    Returns (command, use_shell) suitable for subprocess.run.
    """
    if env_script is None:
        return cmd, False
    cmd_str = subprocess.list2cmdline(cmd)
    if is_windows():
        return f'call "{env_script}" >nul 2>&1 && {cmd_str}', True
    return f'source "{env_script}" >/dev/null 2>&1 && {cmd_str}', True


def _run_executable(
    exe_path: Path,
    args: list[str],
    context: dict[str, Any],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run an executable inside the appropriate Conan env script.

    For native builds, prefers the conanrun env script (always fresh after a
    build) and falls back to the runtime_deploy directory (flat DLL copy for
    CI test jobs where conan install wasn't run).

    For Emscripten builds, uses emrun (browser with logging) for interactive
    launches and Node.js for headless/captured output (tests).
    """
    build_dir = Path(context["build_dir"])
    is_emscripten = exe_path.suffix.lower() in (".js", ".html")

    # Resolve environment: prefer conanrun script for native builds (always
    # fresh), fall back to runtime_deploy dir (for CI without conan install)
    env_script: Path | None = None
    extra_env: dict[str, str] = {}
    if not is_emscripten:
        env_script = _resolve_env_script(build_dir, is_emscripten=False)
        if not env_script:
            runtime_dir = _resolve_runtime_deploy_dir(build_dir)
            if runtime_dir:
                logger.debug(f"Using runtime_deploy: {runtime_dir}")
                path_sep = ";" if is_windows() else ":"
                extra_env["PATH"] = f"{runtime_dir}{path_sep}{os.environ.get('PATH', '')}"
    else:
        env_script = _resolve_env_script(build_dir, is_emscripten=True)

    if is_emscripten and not capture_output:
        html_path = exe_path.with_suffix(".html") if exe_path.suffix.lower() != ".html" else exe_path
        logger.info(f"Launching {html_path.name} with emrun")
        cmd = [sys.executable, "emrun", str(html_path)] + args
    elif is_emscripten:
        js_path = exe_path.with_suffix(".js") if exe_path.suffix.lower() == ".html" else exe_path
        logger.info(f"Running {js_path.name} with Node.js")
        cmd = ["node", "--experimental-wasm-threads", str(js_path)] + args
    else:
        cmd = [str(exe_path)] + args

    run_cmd, use_shell = _shell_wrap(cmd, env_script)

    # Merge extra_env into the process environment
    run_env = None
    if extra_env:
        run_env = {**os.environ, **extra_env}

    try:
        if capture_output:
            return subprocess.run(
                run_cmd, shell=use_shell, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", env=run_env,
            )
        return subprocess.run(run_cmd, shell=use_shell, env=run_env)
    except KeyboardInterrupt:
        sys.exit(0)


def _can_run(context: dict[str, Any]) -> bool:
    """Check if executables can be run on this host."""
    build_dir = Path(context["build_dir"])
    if context["platform"] == "emscripten":
        # Node.js is required for headless WASM execution.
        # Prefer the conanbuild script (provides emsdk tools including node),
        # but also accept a system-installed node (e.g. CI runners).
        if _resolve_env_script(build_dir, is_emscripten=True) is not None:
            return True
        return shutil.which("node") is not None
    if _resolve_env_script(build_dir, is_emscripten=False) is not None:
        return True
    if _resolve_runtime_deploy_dir(build_dir) is not None:
        return True
    # Inline check: can only run natively if target matches host
    return context["platform"] == detect_platform_identifier()


def _run_tests(context: dict[str, Any], verbose: bool) -> int:
    """Run all test executables and return exit code."""
    build_dir = Path(context["build_dir"])
    is_emscripten = context["platform"] == "emscripten"
    test_dir = build_dir / "bin" / "tests"
    logs_dir = Path(context["logs_root"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    test_executables = _discover_executables(test_dir, is_emscripten)
    if not test_executables:
        logger.error(f"No test executables found in: {test_dir}")
        logger.info("Build the project first: ./repo build")
        return 1

    logger.info(f"Found {len(test_executables)} test executable(s)")

    passed = 0
    failed = 0
    failed_tests = []

    for test_exe in sorted(test_executables):
        test_name = test_exe.stem
        log_file = logs_dir / f"test_{test_name}.log"
        test_args = ["--verbose"] if verbose else []

        with log_section(f"Test: {test_name}"):
            try:
                result = _run_executable(test_exe, test_args, context, capture_output=True)

                with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Test: {test_name}\n")
                    f.write(f"Executable: {test_exe}\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write("=" * 70 + "\n")
                    f.write(result.stdout or "")

                if result.stdout:
                    sys.stdout.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        sys.stdout.write("\n")

                if result.returncode == 0:
                    logger.info(f"PASSED: {test_name}")
                    passed += 1
                else:
                    logger.error(f"FAILED: {test_name} (exit code: {result.returncode})")
                    failed += 1
                    failed_tests.append(test_name)

            except Exception as e:
                logger.error(f"FAILED: {test_name} (exception: {e})")
                with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Test: {test_name}\nException: {e}\n")
                failed += 1
                failed_tests.append(test_name)

    with log_section("Test summary"):
        logger.info(f"Total:  {passed + failed}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")

        if failed > 0:
            logger.error("Failed tests:")
            for name in failed_tests:
                logger.error(f"  - {name}")
            return 1
        logger.info("All tests passed!")
    return 0


class LaunchTool(RepoTool):
    name = "launch"
    help = "Launch executables"

    def setup(self, cmd: click.Command) -> click.Command:
        cmd = click.argument(
            "executable",
            required=False,
            default=None,
        )(cmd)
        cmd = click.option(
            "-c", "--config",
            type=click.Choice(
                ["debug", "release", "relwithdebinfo", "minsizerel"],
                case_sensitive=False,
            ),
            default=None,
            help="Build configuration (overrides --build-type)",
        )(cmd)
        cmd = click.option(
            "--env",
            multiple=True,
            help="Environment override (KEY=VALUE). Repeatable.",
        )(cmd)
        cmd = click.option(
            "-i", "--interactive",
            is_flag=True,
            default=None,
            help="Interactive menu to select executable",
        )(cmd)
        return cmd

    def default_args(self, tokens: dict[str, str]) -> dict[str, Any]:
        return {
            "executable": "editor",
            "config": None,
            "env": (),
            "interactive": False,
        }

    def execute(self, ctx: ToolContext, args: dict[str, Any]) -> None:
        root = ctx.workspace_root

        # Support both --config (tool-specific) and --build-type (group-level).
        # --config takes precedence when explicitly provided.
        config_val = args.get("config")
        if config_val:
            build_type = normalize_build_type(config_val)
        else:
            build_type = ctx.dimensions.get("build_type", "Debug")

        platform_id = ctx.dimensions.get("platform", "")
        is_emscripten = platform_id == "emscripten"

        # Build a minimal context dict from tokens for helper functions
        context: dict[str, Any] = {
            "workspace_root": str(root),
            "build_dir": ctx.tokens["build_dir"],
            "platform": platform_id,
            "build_type": build_type,
            "logs_root": ctx.tokens["logs_root"],
        }

        build_dir = Path(context["build_dir"])

        # Apply --env overrides to the process environment so they propagate
        # through the shell-wrapped command.
        env_val = args.get("env")
        if env_val:
            for item in env_val:
                text = str(item).strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    os.environ[key] = value

        # Check if we can run
        if not _can_run(context):
            if is_emscripten:
                logger.error("emsdk not found. Build with --platform emscripten first.")
            else:
                logger.error(f"Cannot run {context['platform']} binaries on this host")
                logger.info(f"Host platform: {detect_platform_identifier()}")
            sys.exit(1)

        # Run single executable
        bin_dir = build_dir / "bin"
        exe_paths = _discover_executables(bin_dir, is_emscripten)

        if not exe_paths:
            logger.error(f"No executables found in: {bin_dir}")
            logger.info("Build the project first: ./repo build")
            sys.exit(1)

        # Interactive mode
        if args.get("interactive"):
            target_exe = _interactive_select(exe_paths)
            if target_exe is None:
                logger.info("No executable selected.")
                sys.exit(0)
        else:
            executable_name = args.get("executable") or "editor"
            target_exe = None
            for exe in exe_paths:
                if exe.stem == executable_name:
                    target_exe = exe
                    break

        if target_exe is None:
            executable_name = args.get("executable") or "editor"
            logger.error(f"Executable not found: {executable_name}")
            logger.info("Available executables:")
            for exe in exe_paths:
                logger.info(f"  {exe.stem}")
            sys.exit(1)

        result = _run_executable(target_exe, ctx.passthrough_args, context)
        sys.exit(result.returncode)
