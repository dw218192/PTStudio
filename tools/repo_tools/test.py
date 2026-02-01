"""Test subcommand implementation."""

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
    is_windows,
    load_repo_config,
    logger,
    normalize_env_config,
    print_subprocess_line,
    print_tool,
    resolve_env_vars,
)


class TestTool(RepoTool):
    name = "test"
    help = "Run unit tests"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--build-type",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration type (default: Debug)",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Verbose test output",
        )
        parser.add_argument(
            "--env",
            action="append",
            help="Environment override (KEY=VALUE). Repeatable.",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(build_type=context["build_type"], verbose=False, env=None)

    def execute(self, args: argparse.Namespace) -> None:
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

        # Find all test executables
        if is_windows():
            test_executables = list(test_dir.glob("*.exe"))
        else:
            # On Unix, find all executable files
            test_executables = [
                f
                for f in test_dir.iterdir()
                if f.is_file() and os.access(f, os.X_OK)
            ]

        if not test_executables:
            print_tool(f"No test executables found in: {test_dir}")
            return

        logger.info(f"Found {len(test_executables)} test executable(s)")

        config_args = get_repo_tool_config_args(config, "launch")
        env_config = normalize_env_config(config_args.get("env"))
        if isinstance(getattr(args, "env", None), list):
            env_overrides = normalize_env_config(args.env)
            env_config.update(env_overrides)
        elif args.env is not None:
            env_config = normalize_env_config(args.env)

        env_vars = resolve_env_vars(env_config, context)
        env = apply_env_overrides(os.environ.copy(), env_vars)

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
            cmd = None

            try:
                # Run the test with verbose output if requested
                cmd = [str(test_exe)]
                if args.verbose:
                    cmd.append("--verbose")

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
                    logger.error(
                        f"✗ {test_name} FAILED (exit code: {result.returncode})"
                    )
                    logger.error(f"  Log: {log_file}")
                    failed += 1
                    failed_tests.append(test_name)

            except Exception as e:
                logger.error(f"✗ {test_name} FAILED (exception: {e})")
                logger.error(f"  Log: {log_file}")
                # Write exception to log file
                cmd_str = ' '.join(cmd) if cmd is not None else "N/A"
                with open(log_file, "w", encoding="utf-8", errors="replace") as f:
                    f.write(f"Test: {test_name}\n")
                    f.write(f"Command: {cmd_str}\n")
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


