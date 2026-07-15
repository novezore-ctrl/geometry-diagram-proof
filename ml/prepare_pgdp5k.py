"""Convert PGDP5K primitive annotations to image/mask pairs.

PGDP5K supplies exact point, line and circle geometry locations. The source
annotations do not expose a reliable per-line dash style in the compact
``geos`` records, so this converter preserves geometry first and marks all
annotated lines as solid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def xy(loc):
    # points/line endpoints are stored as [[x, y]] in PGDP5K.
    if isinstance(loc, list) and loc and isinstance(loc[0], list):
        loc = loc[0]
    return float(loc[0]), float(loc[1])


def one(item, image_path: Path, out_image: Path, out_mask: Path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    geos = item.get("geos") or {}
    width_px = max(2, round(min(width, height) / 180))
    for line in geos.get("lines") or []:
        loc = line.get("loc") or []
        if len(loc) >= 2:
            draw.line([xy(loc[0]), xy(loc[1])], fill=1, width=width_px)
    for circle in geos.get("circles") or []:
        loc = circle.get("loc") or []
        if len(loc) >= 2:
            center = xy(loc[0])
            radius = float(loc[1])
            draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], outline=3, width=width_px)
    point_radius = max(2, round(min(width, height) / 75))
    for point in geos.get("points") or []:
        x, y = xy(point.get("loc") or [])
        draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=4)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image)
    mask.save(out_mask)
    return {"id": image_path.stem, "width": width, "height": height, "points": len(geos.get("points") or []), "lines": len(geos.get("lines") or []), "circles": len(geos.get("circles") or [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="extracted PGDP5K/ directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for split in ("train", "val", "test"):
        annotation = args.root / "annotations" / f"{split}.json"
        data = json.loads(annotation.read_text(encoding="utf-8"))
        for name, item in data.items():
            image_path = args.root / split / item["file_name"]
            if image_path.exists():
                records.append({"split": split, **one(item, image_path, args.out / "images" / f"{split}_{image_path.name}", args.out / "masks" / f"{split}_{image_path.name}")})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"processed={len(records)}")


if __name__ == "__main__":
    main()
