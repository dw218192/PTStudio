"""Process helpers."""

from __future__ import annotations

import subprocess
import sys


def run(*cmd: str, **kwargs) -> int:
    """Echo + run a command, returning its exit code."""
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", flush=True)
    return subprocess.call([str(c) for c in cmd], **kwargs)


def run_or_exit(*cmd: str, **kwargs) -> None:
    """run(), but exit this process with the command's code on failure."""
    code = run(*cmd, **kwargs)
    if code:
        sys.exit(code)
