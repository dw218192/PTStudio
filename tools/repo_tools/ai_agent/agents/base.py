"""Base interfaces for agent runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class AgentRunResult:
    returncode: int
    stdout: str
    stderr: str


class AgentRunner(Protocol):
    def run_step(
        self,
        prompt: str,
        context_dir: Path,
        output_file: Path,
        *,
        skill: str | None = None,
        subagent: str | None = None,
    ) -> AgentRunResult:
        """Run a single agent step."""
