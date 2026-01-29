"""Agent command entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from repo_tools import load_repo_config
from repo_tools.ai_agent.agents.cursor_cli import CursorCliAgent
from repo_tools.ai_agent.config import load_agent_config
from repo_tools.ai_agent.orchestrator import run_agent_loop


def agent_command(args: argparse.Namespace) -> None:
    root = Path(__file__).parent.parent.parent
    repo_config = load_repo_config(root)
    config = load_agent_config(root, repo_config, human_review_mode=args.human_review_mode)

    prd_path = None
    if args.prd:
        prd_path = Path(args.prd)
        if not prd_path.is_absolute():
            prd_path = root / prd_path
        prd_path = str(prd_path)

    agent = CursorCliAgent(config.cursor_command_template, config.workspace_root)
    run_agent_loop(agent=agent, config=config, goal=args.goal or "", prd_path=prd_path)


def register_agent_command(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("agent", help="Run autonomous agent loop")
    parser.add_argument("--goal", type=str, help="Optional goal for the agent loop")
    parser.add_argument("--prd", type=str, help="Path to PRD file")
    parser.add_argument(
        "--human-review-mode",
        choices=["strict", "balanced", "auto"],
        help="Human review threshold (overrides config)",
    )
    parser.set_defaults(func=agent_command)
