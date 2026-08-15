#!/usr/bin/env python3
"""Stitch 2+ images into a horizontal or vertical grid, with optional
per-tile filename annotations.

Usage: imggrid.py IMG IMG [IMG ...] -o OUT [-a] [--direction h|v] [--labels a,b,c]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    # Try a few common fonts; fall back to Pillow default so the script
    # works even on a minimal install. default_font is bitmap-only; the
    # truetype path gives readable sizes.
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _annotate(img: Image.Image, text: str) -> Image.Image:
    """Overlay *text* on the top-left of *img* with a dark backdrop."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(max(14, min(out.size) // 40))
    # Backdrop sized to the rendered text.
    pad = 4
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
    except AttributeError:
        # Very old Pillow. textsize removed in newer versions.
        tw, th = draw.textsize(text, font=font)
    draw.rectangle((0, 0, tw + 2 * pad, th + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Stitch images into a grid.")
    ap.add_argument("images", nargs="+", type=Path, help="Input images (>=2)")
    ap.add_argument("-o", "--output", required=True, type=Path,
                    help="Output PNG path")
    ap.add_argument("-a", "--annotate", action="store_true",
                    help="Overlay the basename (or --labels) on each tile")
    ap.add_argument("--direction", choices=("horizontal", "vertical"),
                    default="horizontal",
                    help="Stitch direction (default: horizontal)")
    ap.add_argument("--labels", default=None,
                    help="Comma-separated labels, one per image")
    args = ap.parse_args(argv)

    if len(args.images) < 2:
        print("imggrid: need at least 2 images", file=sys.stderr)
        return 2

    for p in args.images:
        if not p.exists():
            print(f"imggrid: input not found: {p}", file=sys.stderr)
            return 2

    labels: list[str]
    if args.labels:
        labels = args.labels.split(",")
        if len(labels) != len(args.images):
            print(
                f"imggrid: --labels has {len(labels)} entries but got "
                f"{len(args.images)} images",
                file=sys.stderr,
            )
            return 2
    else:
        labels = [p.name for p in args.images]

    tiles: list[Image.Image] = []
    for path, label in zip(args.images, labels):
        img = Image.open(path).convert("RGB")
        if args.annotate:
            img = _annotate(img, label)
        tiles.append(img)

    if args.direction == "horizontal":
        out_h = max(t.size[1] for t in tiles)
        out_w = sum(t.size[0] for t in tiles)
        out = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        x = 0
        for t in tiles:
            out.paste(t, (x, 0))
            x += t.size[0]
    else:
        out_w = max(t.size[0] for t in tiles)
        out_h = sum(t.size[1] for t in tiles)
        out = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        y = 0
        for t in tiles:
            out.paste(t, (0, y))
            y += t.size[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"imggrid: wrote {args.output} ({out.size[0]}x{out.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
