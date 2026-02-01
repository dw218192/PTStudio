"""Repo paths subcommand implementation."""

import argparse
import json
from pathlib import Path

from repo_tools import (
    RepoContext,
    RepoTool,
    build_repo_context,
    load_repo_config,
    print_tool,
)


class RepoPathsTool(RepoTool):
    name = "repo_paths"
    help = "Show resolved repository paths"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--build-type",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration type (default: Debug)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output resolved paths as JSON",
        )

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(build_type=context["build_type"], json=False)

    def execute(self, args: argparse.Namespace) -> None:
        root = Path(__file__).parent.parent.parent
        config = load_repo_config(root)
        context = build_repo_context(root, args.build_type, config)
        if args.json:
            print(json.dumps(context, indent=4))
            return
        for key in sorted(context.keys()):
            print_tool(f"{key}: {context[key]}")


