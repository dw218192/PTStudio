"""Cursor CLI agent runner implementation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from repo_tools import apply_token_replacements, logger
from repo_tools.ai_agent.agents.base import AgentRunResult, AgentRunner


class CursorCliAgent(AgentRunner):
    def __init__(self, command_template: list[str], workspace_root: Path) -> None:
        self._command_template = command_template
        self._workspace_root = workspace_root

    def run_step(
        self,
        prompt: str,
        context_dir: Path,
        output_file: Path,
        *,
        skill: str | None = None,
        subagent: str | None = None,
    ) -> AgentRunResult:
        context_dir.mkdir(parents=True, exist_ok=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        prompt_file = context_dir / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        replacements = {
            "workspace_root": str(self._workspace_root),
            "prompt_file": str(prompt_file),
            "context_dir": str(context_dir),
            "output_file": str(output_file),
            "skill": skill or "",
            "subagent": subagent or "",
        }

        command = apply_token_replacements(self._command_template, replacements)

        logger.info(f"Running agent command: {' '.join(command)}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )

        if not output_file.exists() and result.stdout.strip():
            output_file.write_text(result.stdout, encoding="utf-8")

        return AgentRunResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
