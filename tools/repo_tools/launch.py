"""Launch subcommand implementation - runs executables and tests."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from repo_tools import (
    RepoContext,
    RepoTool,
    apply_env_overrides,
    build_repo_context,
    get_repo_tool_config_args,
    is_platform_compatible,
    is_windows,
    load_repo_config,
    log_section,
    logger,
    normalize_build_type,
    normalize_env_config,
    resolve_env_vars,
)

def _get_node_path(context: RepoContext) -> Path | None:
    """Get path to Node.js bundled inside the emsdk Conan package."""
    build_deps = Path(context["build_deps_root"])
    emsdk_dir = build_deps / "emsdk"

    if not emsdk_dir.exists():
        logger.debug(f"emsdk directory not found: {emsdk_dir}")
        return None

    # Navigate: emsdk/<version>/<arch>/bin/node/<node_version>/bin/node[.exe]
    for version_dir in emsdk_dir.iterdir():
        if not version_dir.is_dir():
            continue
        for arch_dir in version_dir.iterdir():
            if not arch_dir.is_dir():
                continue
            node_base = arch_dir / "bin" / "node"
            if not node_base.is_dir():
                continue
            for node_ver_dir in node_base.iterdir():
                if not node_ver_dir.is_dir():
                    continue
                node_exe = node_ver_dir / "bin" / ("node.exe" if is_windows() else "node")
                if node_exe.exists():
                    return node_exe

    logger.warning("Node.js not found in emsdk package")
    return None


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
    return exe_paths


def _setup_environment(context: RepoContext, config: dict, env_overrides: dict | None) -> dict:
    """Set up environment variables for running executables."""
    config_args = get_repo_tool_config_args(config, "launch")
    env_config = normalize_env_config(config_args.get("env"))
    if env_overrides:
        env_config.update(env_overrides)
    env_vars = resolve_env_vars(env_config, context)
    return apply_env_overrides(os.environ.copy(), env_vars)


def _run_executable(
    exe_path: Path,
    args: list[str],
    env: dict,
    context: RepoContext,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run an executable.

    For Emscripten builds, uses emrun (browser with logging) for interactive
    launches and Node.js for headless/captured output (tests).
    """
    is_emscripten = exe_path.suffix.lower() in (".js", ".html")

    if is_emscripten and not capture_output:
        # Interactive launch: use emrun (serves files, opens browser, captures logs)
        emrun = _get_emrun_path(context)
        if emrun is None:
            raise RuntimeError("emrun not found. Build with --platform emscripten first.")
        html_path = exe_path.with_suffix(".html") if exe_path.suffix.lower() != ".html" else exe_path
        logger.info(f"Launching {html_path.name} with emrun")
        cmd = [sys.executable, str(emrun), str(html_path)] + args
    elif is_emscripten:
        # Headless (tests): use Node.js
        node_path = _get_node_path(context)
        if node_path is None:
            raise RuntimeError("Node.js not found. Build with --platform emscripten first.")
        js_path = exe_path.with_suffix(".js") if exe_path.suffix.lower() == ".html" else exe_path
        logger.info(f"Running {js_path.name} with Node.js")
        cmd = [str(node_path), "--experimental-wasm-threads", str(js_path)] + args
    else:
        cmd = [str(exe_path)] + args

    try:
        if capture_output:
            return subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
            )
        return subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        sys.exit(0)


def _can_run(context: RepoContext) -> bool:
    """Check if executables can be run on this host."""
    if context["platform"] == "emscripten":
        return _get_emrun_path(context) is not None or _get_node_path(context) is not None
    return is_platform_compatible(context["platform"])


def _get_emrun_path(context: RepoContext) -> Path | None:
    """Get path to emrun bundled inside the emsdk Conan package."""
    build_deps = Path(context["build_deps_root"])
    emsdk_dir = build_deps / "emsdk"

    if not emsdk_dir.exists():
        logger.debug(f"emsdk directory not found: {emsdk_dir}")
        return None

    # Navigate: emsdk/<version>/<arch>/bin/upstream/emscripten/emrun.py
    for version_dir in emsdk_dir.iterdir():
        if not version_dir.is_dir():
            continue
        for arch_dir in version_dir.iterdir():
            if not arch_dir.is_dir():
                continue
            emrun = arch_dir / "bin" / "upstream" / "emscripten" / "emrun.py"
            if emrun.exists():
                return emrun

    logger.warning("emrun not found in emsdk package")
    return None



def _run_tests(context: RepoContext, config: dict, env: dict, verbose: bool) -> int:
    """Run all test executables and return exit code."""
    build_dir = Path(context["build_dir"])
    is_emscripten = context["platform"] == "emscripten"
    test_dir = build_dir / "bin" / "tests"
    logs_dir = Path(context["logs_root"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    test_executables = _discover_executables(test_dir, is_emscripten)
    if not test_executables:
        logger.error(f"No test executables found in: {test_dir}")
        logger.info("Build the project first: ./pts build")
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
                result = _run_executable(test_exe, test_args, env, context, capture_output=True)

                with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Test: {test_name}\n")
                    f.write(f"Executable: {test_exe}\n")
                    f.write(f"Exit code: {result.returncode}\n")
                    f.write("=" * 70 + "\n")
                    f.write(result.stdout or "")

                if result.stdout:
                    for line in result.stdout.splitlines():
                        logger.info(f"  {line}")

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
    help = "Launch executables or run tests"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "executable", type=str, nargs="?", default=argparse.SUPPRESS,
            help="Executable to launch (default: editor)",
        )
        parser.add_argument(
            "-c", "--config", type=str.casefold,
            choices=["debug", "release", "relwithdebinfo", "minsizerel"],
            help="Build configuration (default: debug)",
        )
        parser.add_argument(
            "--build-type",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Alias for --config",
        )
        parser.add_argument(
            "--env", action="append",
            help="Environment override (KEY=VALUE). Repeatable.",
        )
        parser.add_argument(
            "--test", action="store_true",
            help="Run all test executables",
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true",
            help="Verbose output (for tests)",
        )
        parser.add_argument(
            "-i", "--interactive", action="store_true",
            help="Interactive menu to select executable",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(
            platform=context["platform"],
            executable="editor",
            config=context["build_type"].casefold(),
            build_type=None,
            env=None,
            test=False,
            verbose=False,
            interactive=False,
        )

    def execute(self, args: argparse.Namespace) -> None:
        root = Path(__file__).parent.parent.parent
        # Support both --config and --build-type
        build_type = normalize_build_type(args.build_type or args.config)
        config = load_repo_config(root)
        context = build_repo_context(root, build_type, config, args.platform)
        build_dir = Path(context["build_dir"])
        is_emscripten = context["platform"] == "emscripten"

        env_overrides = normalize_env_config(args.env) if args.env else None
        env = _setup_environment(context, config, env_overrides)

        # Check if we can run
        if not _can_run(context):
            from repo_tools import detect_platform_identifier
            if is_emscripten:
                logger.error("emsdk not found. Build with --platform emscripten first.")
            else:
                logger.error(f"Cannot run {context['platform']} binaries on this host")
                logger.info(f"Host platform: {detect_platform_identifier()}")
            sys.exit(1)

        # Run tests
        if args.test:
            sys.exit(_run_tests(context, config, env, args.verbose))

        # Run single executable
        bin_dir = build_dir / "bin"
        exe_paths = _discover_executables(bin_dir, is_emscripten)

        if not exe_paths:
            logger.error(f"No executables found in: {bin_dir}")
            logger.info("Build the project first: ./pts build")
            sys.exit(1)

        # Interactive mode
        if args.interactive:
            target_exe = _interactive_select(exe_paths)
            if target_exe is None:
                logger.info("No executable selected.")
                sys.exit(0)
        else:
            target_exe = None
            for exe in exe_paths:
                if exe.stem == args.executable:
                    target_exe = exe
                    break

        if target_exe is None:
            logger.error(f"Executable not found: {args.executable}")
            logger.info("Available executables:")
            for exe in exe_paths:
                logger.info(f"  {exe.stem}")
            sys.exit(1)

        result = _run_executable(target_exe, args.passthrough_args, env, context)
        sys.exit(result.returncode)
