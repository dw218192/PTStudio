#!/usr/bin/env python3
"""`pixi run lint` -- check formatting without modifying files.

Delegates to the repo tooling CLI; see tasks/repo.py.
Extra arguments are forwarded, e.g.:

    pixi run lint --platform emscripten --build-type Release
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *"format --verify".split(), *sys.argv[1:]]
    sys.exit(main())
