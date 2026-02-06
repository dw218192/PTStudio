"""Agent tool - launches coding agents with repo-specific config."""

import argparse
import sys

from .. import RepoTool
from .runner import AgentRunner
from .claude import ClaudeRunner


RUNNERS: dict[str, AgentRunner] = {
    "claude": ClaudeRunner(),
}


class AgentTool(RepoTool):
    name = "agent"
    help = "Run coding agents with workflows tailored for this repository."

    def setup(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="agent", required=True)
        for name, runner in RUNNERS.items():
            sub = subparsers.add_parser(name, help=runner.help)
            runner.setup(sub)

    def execute(self, args: argparse.Namespace) -> None:
        runner = RUNNERS.get(args.agent)
        if runner:
            sys.exit(runner.run(args))
        raise ValueError(f"Unknown agent: {args.agent}")
