#!/usr/bin/env python3
"""Crop an image to a box.

Usage: imgcrop.py IMG --box X,Y,W,H -o OUT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _parse_box(s: str) -> tuple[int, int, int, int]:
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--box expects X,Y,W,H (4 ints); got {s!r}"
        )
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--box ints only: {e}") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            f"--box width and height must be > 0 (got {w}x{h})"
        )
    return x, y, w, h


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Crop an image to a box.")
    ap.add_argument("image", type=Path, help="Input image path")
    ap.add_argument("--box", required=True, type=_parse_box,
                    help="Crop box: X,Y,W,H in pixels")
    ap.add_argument("-o", "--output", required=True, type=Path,
                    help="Output PNG path")
    args = ap.parse_args(argv)

    if not args.image.exists():
        print(f"imgcrop: input not found: {args.image}", file=sys.stderr)
        return 2

    x, y, w, h = args.box
    with Image.open(args.image) as img:
        # Clip to image bounds -- Pillow crop() just pads with black beyond
        # edges, which hides "bbox was wrong" bugs. Explicitly reject.
        iw, ih = img.size
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            print(
                f"imgcrop: box ({x},{y},{w},{h}) outside image {iw}x{ih}",
                file=sys.stderr,
            )
            return 2
        cropped = img.crop((x, y, x + w, y + h))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(args.output)
    print(f"imgcrop: wrote {args.output} ({w}x{h})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
