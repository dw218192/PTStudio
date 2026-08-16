#!/usr/bin/env python3
"""`pixi run test` -- run the test suite.

Delegates to the repo tooling CLI; see tasks/repo.py.
Extra arguments are forwarded, e.g.:

    pixi run test --platform emscripten --build-type Release
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *"test".split(), *sys.argv[1:]]
    sys.exit(main())
