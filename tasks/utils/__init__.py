"""Shared helpers for `tasks/*.py`.

pixi puts `tasks/` first on sys.path when running a task, so scripts can
`from utils import run`.
"""

from .proc import run

__all__ = ["run"]
