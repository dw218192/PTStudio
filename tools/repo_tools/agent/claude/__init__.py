"""Claude Code backend.

Prompt format::

    ────────────────────────   (horizontal rule)
    <command type>
    <command string>           (may be multi-line)
    <description>              (last non-empty line before marker)
    Do you want to proceed?
     ❯ 1. Yes
       2. Yes, and don't ask again for <...>
       3. No
"""

from __future__ import annotations

from pathlib import Path

from ..runner import AgentCLITool, ToolRequest

_PROMPT_MARKER = "Do you want to proceed?"
_ALLOWED_TOOLS = ["Edit", "Write"]

def _detect_prompt(pane_text: str) -> ToolRequest | None:
    """Return a :class:`ToolRequest` when a Claude Code permission
    prompt is visible, or ``None`` otherwise."""
    lines = pane_text.splitlines()

    marker_idx: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if _PROMPT_MARKER in lines[i]:
            marker_idx = i
            break
    if marker_idx is None:
        return None

    block: list[str] = []
    for i in range(marker_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped and all(c in "─━—-" for c in stripped):
            break
        if stripped:
            # Preserve trailing whitespace (needed to undo terminal
            # line-wrapping correctly — a trailing space means the wrap
            # point was at a space, not mid-word).
            block.append(lines[i].lstrip())

    block.reverse()

    if len(block) < 2:
        return None

    tool = block[0].split()[0]
    command = "\n".join(block[1:-1]) if len(block) > 2 else None
    return ToolRequest(tool, command)


class Claude(AgentCLITool):
    """Launch Claude Code with repo-specific config."""

    def build_command(self, cwd: str | None = None) -> list[str]:  # noqa: ARG002
        config_dir = Path(__file__).parent
        sys_prompt = config_dir / "claude_sys.txt"
        return [
            "claude",
            "--allowedTools", *_ALLOWED_TOOLS,
            "--append-system-prompt",
            sys_prompt.read_text(encoding="utf-8"),
        ]

    def detect_prompt(self, pane_text: str) -> ToolRequest | None:
        return _detect_prompt(pane_text)

    @property
    def approve_key(self) -> str:
        return "\r"

    @property
    def deny_key(self) -> str:
        return "\x1b"  # Escape
