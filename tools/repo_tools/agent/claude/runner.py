"""Claude Code runner."""

import argparse
import subprocess
from pathlib import Path

from ..runner import AgentRunner


class ClaudeRunner(AgentRunner):
    name = "claude"
    help = "Launch Claude Code with repo-specific config"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        pass  # No additional args; passthrough handled by tool

    def run(self, args: argparse.Namespace) -> int:
        config_dir = Path(__file__).parent
        sys_prompt = config_dir / "claude_sys.txt"
        settings = config_dir / "settings.json"

        cmd = [
            "claude",
            "--append-system-prompt", sys_prompt.read_text(encoding="utf-8"),
            "--settings", str(settings),
        ]
        if hasattr(args, "passthrough_args") and args.passthrough_args:
            cmd.extend(args.passthrough_args)

        return subprocess.run(cmd).returncode
