"""Image-diff tooling: golden-image regression via FLIP.

Exports two RepoTools discovered by the framework:

- ``image-diff`` — run renderer captures and diff against committed GT.
- ``bake-gt``    — bake path-traced GT PNGs into the configured gt_dir.
"""

from __future__ import annotations

from .bake import BakeGtTool
from .diff import ImageDiffTool

__all__ = ["BakeGtTool", "ImageDiffTool"]
