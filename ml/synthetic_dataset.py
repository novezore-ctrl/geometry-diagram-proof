"""Generate simple geometry images and pixel masks for pipeline smoke tests."""
from __future__ import annotations

import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw

W = H = 256


def one(out: Path, index: int) -> None:
    image = Image.new("RGB", (W, H), "white")
    mask = Image.new("L", (W, H), 0)
    draw, label = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    width = random.choice((2, 3, 4))
    pts = [(random.randint(30, 70), random.randint(170, 220)), (random.randint(105, 145), random.randint(25, 75)), (random.randint(185, 225), random.randint(170, 220))]
    draw.line((*pts[0], *pts[1]), fill="black", width=width); label.line((*pts[0], *pts[1]), fill=1, width=width)
    draw.line((*pts[1], *pts[2]), fill="black", width=width); label.line((*pts[1], *pts[2]), fill=1, width=width)
    draw.line((*pts[0], *pts[2]), fill="black", width=width); label.line((*pts[0], *pts[2]), fill=1, width=width)
    if random.random() < .7:
        cx, cy, r = 128, random.randint(90, 160), random.randint(25, 55)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="black", width=width); label.ellipse((cx-r, cy-r, cx+r, cy+r), outline=3, width=width)
    for x, y in pts:
        draw.ellipse((x-4, y-4, x+4, y+4), fill="black"); label.ellipse((x-4, y-4, x+4, y+4), fill=4)
    image.save(out / f"{index:05d}.png")
    mask.save(out / f"{index:05d}.mask.png")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count): one(args.out, i)
    print(f"generated {args.count} image/mask pairs in {args.out}")


if __name__ == "__main__": main()
