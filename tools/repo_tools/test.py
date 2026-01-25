"""Test subcommand implementation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from repo_tools import (
    apply_env_overrides,
    build_repo_context,
    compile_slang_shaders,
    is_windows,
    load_repo_config,
    logger,
    print_subprocess_line,
    print_tool,
    resolve_env_vars,
)


def test_command(args: argparse.Namespace) -> None:
    """Test subcommand implementation."""
    root = Path(__file__).parent.parent.parent
    config = load_repo_config(root)
    context = build_repo_context(root, args.build_type, config)
    build_dir = Path(context["build_dir"])
    test_dir = build_dir / "bin" / "tests"
    logs_dir = Path(context["logs_root"])

    # Create logs directory
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        print_tool(f"Test directory does not exist: {test_dir}")
        print_tool("Build the project first with: pts.cmd build")
        sys.exit(1)

    compile_slang_shaders(root, config, context, logs_dir)

    # Find all test executables
    if is_windows():
        test_executables = list(test_dir.glob("*.exe"))
    else:
        # On Unix, find all executable files
        test_executables = [
            f for f in test_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)
        ]

    if not test_executables:
        print_tool(f"No test executables found in: {test_dir}")
        return

    logger.info(f"Found {len(test_executables)} test executable(s)")

    # Track test results
    passed = 0
    failed = 0
    failed_tests = []

    # Run each test
    for test_exe in sorted(test_executables):
        test_name = test_exe.stem
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'=' * 70}")

        # Create log file for this test
        log_file = logs_dir / f"test_{test_name}.log"

        try:
            # Run the test with verbose output if requested
            cmd = [str(test_exe)]
            if args.verbose:
                cmd.append("--verbose")
            env_vars = resolve_env_vars(config, context)
            env = apply_env_overrides(os.environ.copy(), env_vars)

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            # Write output to log file
            with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write("=" * 70 + "\n")
                f.write(result.stdout)

            # Print output to console
            for line in result.stdout.splitlines():
                print_subprocess_line(line)

            if result.returncode == 0:
                logger.info(f"✓ {test_name} PASSED")
                logger.info(f"  Log: {log_file}")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED (exit code: {result.returncode})")
                logger.error(f"  Log: {log_file}")
                failed += 1
                failed_tests.append(test_name)

        except Exception as e:
            logger.error(f"✗ {test_name} FAILED (exception: {e})")
            logger.error(f"  Log: {log_file}")
            # Write exception to log file
            with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exception: {e}\n")
            failed += 1
            failed_tests.append(test_name)

    # Print summary
    logger.info(f"{'=' * 70}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total tests:  {passed + failed}")
    logger.info(f"Passed:       {passed}")
    logger.info(f"Failed:       {failed}")

    if failed > 0:
        logger.error(f"\nFailed tests:")
        for test_name in failed_tests:
            logger.error(f"  - {test_name}")
        sys.exit(1)
    else:
        logger.info(f"\n✓ All tests passed!")


def register_test_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the test subcommand."""
    parser = subparsers.add_parser("test", help="Run unit tests")
    parser.add_argument(
        "--build-type",
        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
        default="Debug",
        help="Build configuration type (default: Debug)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose test output",
    )
    parser.set_defaults(func=test_command)
