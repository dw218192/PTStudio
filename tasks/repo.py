#!/usr/bin/env python3
"""Dispatch to the repo tooling CLI.

This is the pixi-era replacement for the generated `./repo` shim. The shim
existed to point a bootstrapped venv at `repo_tools.cli`; pixi already provides
the interpreter and the dependencies, so all that is left is putting the two
tool trees on sys.path and handing over argv.

    pixi run repo --help
    pixi run repo build --platform emscripten --build-type Release

Tool implementations are unchanged:
  tools/repo_tools/           project-owned tools (build, slangc, launch, ...)
  tools/framework/repo_tools/ repokit, imported as a plain library
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK = REPO_ROOT / "tools" / "framework"
PROJECT_TOOLS = REPO_ROOT / "tools"


def main() -> int:
    # `repo_tools` is a namespace package split across both trees, so both
    # parents must be importable for tool discovery to see project + framework.
    for path in (str(FRAMEWORK), str(PROJECT_TOOLS)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Child processes (conan, cmake) inherit this; keep discovery consistent.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(FRAMEWORK), str(PROJECT_TOOLS)] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    from repo_tools.cli import main as cli_main

    # The CLI reads --workspace-root from argv; inject it when absent so tasks
    # work from any cwd.
    argv = sys.argv[1:]
    if not any(a == "--workspace-root" or a.startswith("--workspace-root=") for a in argv):
        argv = ["--workspace-root", str(REPO_ROOT)] + argv
    sys.argv = [sys.argv[0]] + argv

    try:
        cli_main()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
