"""Main entry point for repo tools."""

import argparse
import sys
from pathlib import Path

from repo_tools.build import register_build_command
from repo_tools.format import register_format_command
from repo_tools.test import register_test_command


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PTStudio repository tools")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Register tooling commands
    register_build_command(subparsers)
    register_format_command(subparsers)
    register_test_command(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
