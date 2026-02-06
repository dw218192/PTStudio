"""Run Python in the repo tooling environment."""

import argparse
import subprocess
import sys

from repo_tools import RepoTool


class PythonTool(RepoTool):
    name = "python"
    help = "Run Python in the repo tooling environment"

    def execute(self, args: argparse.Namespace) -> None:
        extra = getattr(args, "passthrough_args", [])
        raise SystemExit(subprocess.call([sys.executable, *extra]))
