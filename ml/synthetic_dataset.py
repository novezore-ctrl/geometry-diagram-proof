"""Generate geometry-like images and six-class pixel masks for pretraining."""
from __future__ import annotations

import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W = H = 320
SOLID, DASHED, CURVE, POINT, TEXT = 1, 2, 3, 4, 5


def font():
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try: return ImageFont.truetype(path, random.randint(17, 25))
        except OSError: pass
    return ImageFont.load_default()


def dashed(draw, mask, a, b, width):
    x1, y1 = a; x2, y2 = b; length = max(1, ((x2-x1)**2 + (y2-y1)**2) ** .5); ux, uy = (x2-x1)/length, (y2-y1)/length
    for start in range(0, int(length), 18):
        end = min(start + 10, length)
        p, q = (x1 + ux*start, y1 + uy*start), (x1 + ux*end, y1 + uy*end)
        draw.line((*p, *q), fill="black", width=width); mask.line((*p, *q), fill=DASHED, width=width)


def one(out: Path, index: int) -> None:
    image = Image.new("RGB", (W, H), "white"); mask = Image.new("L", (W, H), 0)
    draw, label = ImageDraw.Draw(image), ImageDraw.Draw(mask); width = random.choice((2, 3, 4))
    pts = [(random.randint(30, 75), random.randint(220, 270)), (random.randint(135, 185), random.randint(35, 85)), (random.randint(245, 290), random.randint(220, 270))]
    edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[0], pts[2])]
    for a, b in edges:
        if random.random() < .3: dashed(draw, label, a, b, width)
        else: draw.line((*a, *b), fill="black", width=width); label.line((*a, *b), fill=SOLID, width=width)
    if random.random() < .8:
        cx, cy = random.randint(120, 200), random.randint(120, 205); rx, ry = random.randint(28, 65), random.randint(22, 55)
        draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), outline="black", width=width); label.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), outline=CURVE, width=width)
    for x, y in pts:
        draw.ellipse((x-4, y-4, x+4, y+4), fill="black"); label.ellipse((x-4, y-4, x+4, y+4), fill=POINT)
    if random.random() < .9:
        text_draw, text_mask = ImageDraw.Draw(image), ImageDraw.Draw(mask); f = font()
        for text, (x, y) in zip(random.sample(("A", "B", "C", "D", "O", "α", "β", "F₁", "F₂"), 3), pts):
            tx, ty = x + random.randint(-18, 14), y + random.choice((-30, 8)); text_draw.text((tx, ty), text, fill="black", font=f); text_mask.text((tx, ty), text, fill=TEXT, font=f)
    image.save(out / f"{index:05d}.png"); mask.save(out / f"{index:05d}.mask.png")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--count", type=int, default=100); args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count): one(args.out, i)
    print(f"generated {args.count} image/mask pairs in {args.out}")


if __name__ == "__main__": main()
