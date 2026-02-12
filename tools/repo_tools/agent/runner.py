"""Abstract base class for CLI-based agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wezterm import PaneSession


class ToolRequest:
    """What an agent is asking permission to use."""

    __slots__ = ("tool", "command")

    def __init__(self, tool: str, command: str | None = None) -> None:
        self.tool = tool
        self.command = command

    def __repr__(self) -> str:
        if self.command:
            return f"{self.tool}({self.command[:80]})"
        return self.tool


class AgentCLITool(ABC):
    """Backend descriptor for a terminal-based coding agent.

    Knows how to build the launch command, detect permission prompts,
    and which keystrokes to send for approve/deny.  Does NOT own the
    session lifecycle — that belongs to the agent tool.
    """

    @abstractmethod
    def build_command(self, cwd: str | None = None) -> list[str]:
        """Return the CLI command to launch this agent."""

    @abstractmethod
    def detect_prompt(self, pane_text: str) -> ToolRequest | None:
        """Return a :class:`ToolRequest` if a permission prompt is
        visible, or ``None`` otherwise."""

    @property
    @abstractmethod
    def approve_key(self) -> str:
        """Keystroke(s) to send to approve the prompt."""

    @property
    @abstractmethod
    def deny_key(self) -> str:
        """Keystroke(s) to send to deny the prompt."""

    def send_text(self, session: PaneSession, text: str) -> None:
        """Paste *text* into the session and submit with Enter."""
        import time

        session.send_text(text)
        time.sleep(0.3)  # let the UI process the paste before submitting
        session.send_keys("\r")
