"""WezTerm CLI integration for agent spawning and pane management."""

from __future__ import annotations

import os
import shutil
import subprocess

from .. import logger


def ensure_installed() -> str:
    """Return path to wezterm, or exit with error.

    Must be run from inside a WezTerm terminal (``WEZTERM_PANE`` set)
    so that ``wezterm cli`` can reach the mux socket.
    """
    path = shutil.which("wezterm")
    if not path:
        logger.error(
            "WezTerm is required but not found on PATH.\n"
            "  Install from: https://wezfurlong.org/wezterm/"
        )
        raise SystemExit(1)
    if not os.environ.get("WEZTERM_PANE"):
        logger.error("This command must be run from inside a WezTerm terminal.")
        raise SystemExit(1)
    return path


def _run_cli(
    *args: str, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    wez = ensure_installed()
    result = subprocess.run(
        [wez, "cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=input_text,
    )
    if result.returncode != 0:
        logger.debug(f"wezterm cli {args[0]} failed: {result.stderr.strip()}")
    return result


class PaneSession:
    """A managed WezTerm pane with get/send/alive operations."""

    def __init__(self, pane_id: int) -> None:
        self.pane_id = pane_id

    def get_text(self) -> str:
        """Get visible text content from the pane."""
        result = _run_cli("get-text", "--pane-id", str(self.pane_id))
        if result.returncode != 0:
            return ""
        return result.stdout

    def send_keys(self, keys: str) -> bool:
        """Send raw keypresses (escape sequences, Enter, etc.)."""
        result = _run_cli(
            "send-text", "--pane-id", str(self.pane_id),
            "--no-paste", input_text=keys,
        )
        return result.returncode == 0

    def send_text(self, text: str) -> bool:
        """Paste text into the pane."""
        result = _run_cli(
            "send-text", "--pane-id", str(self.pane_id),
            input_text=text,
        )
        return result.returncode == 0

    def is_alive(self) -> bool:
        """Return True if the pane still exists."""
        result = _run_cli("list", "--format", "json")
        if result.returncode != 0:
            return False
        import json
        try:
            panes = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False
        return any(p.get("pane_id") == self.pane_id for p in panes)

    def kill(self) -> None:
        """Kill this pane."""
        _run_cli("kill-pane", "--pane-id", str(self.pane_id))

    @classmethod
    def spawn(cls, cmd: list[str], cwd: str | None = None) -> PaneSession | None:
        """Spawn *cmd* in a new WezTerm window, return session or None."""
        cli_args = ["spawn", "--new-window"]
        if cwd:
            cli_args.extend(["--cwd", cwd])
        cli_args.append("--")
        cli_args.extend(cmd)
        result = _run_cli(*cli_args)
        if result.returncode != 0:
            return None
        try:
            return cls(int(result.stdout.strip()))
        except ValueError:
            logger.warning(f"Unexpected spawn output: {result.stdout.strip()}")
            return None
