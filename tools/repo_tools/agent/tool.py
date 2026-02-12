"""Agent tool - launches coding agents with repo-specific config."""

import argparse
import sys
import time
from pathlib import Path

from .. import RepoContext, RepoTool, logger
from .approver import AutoApprover
from .claude import Claude
from .runner import AgentCLITool
from .wezterm import PaneSession, ensure_installed

BACKENDS: dict[str, AgentCLITool] = {
    "claude": Claude(),
}


class AgentTool(RepoTool):
    name = "agent"
    help = "Run coding agents with workflows tailored for this repository."

    def default_args(self, context: RepoContext) -> argparse.Namespace:
        return argparse.Namespace(workspace_root=context["workspace_root"])

    def setup(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="subcommand", required=True)
        run_parser = sub.add_parser("run", help="Run an agent in a managed terminal")
        run_parser.add_argument(
            "--backend",
            choices=BACKENDS,
            default="claude",
            help="Agent backend to use (default: claude)",
        )
        run_parser.add_argument(
            "--auto-approve",
            action="store_true",
            default=False,
            help="Auto-approve tool permissions that match rules.toml",
        )

    def execute(self, args: argparse.Namespace) -> None:
        ensure_installed()
        backend = BACKENDS[args.backend]
        cwd = getattr(args, "workspace_root", None)
        cmd = backend.build_command(cwd=cwd)

        session = PaneSession.spawn(cmd, cwd=cwd)
        if session is None:
            logger.error("Failed to obtain WezTerm pane.")
            sys.exit(1)

        logger.info(f"{args.backend} running in WezTerm pane {session.pane_id}")

        if not getattr(args, "auto_approve", False):
            return

        rules = Path(__file__).parent / "rules.toml"
        approver = AutoApprover(backend, session, rules, project_root=Path(cwd) if cwd else None)
        approver.start()

        try:
            while session.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            approver.stop()
            session.kill()
            logger.info("Session closed.")
