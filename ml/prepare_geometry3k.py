"""Rasterize Geometry3K structural annotations into training masks.

This is intentionally conservative: only point coordinates, named line
instances, and circle centers with inferable radii are rasterized. Text and
dashed-vs-solid line style are not present reliably in the source annotations,
so they remain background and are reported in the manifest.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def parse_line(name: str):
    letters = [c for c in str(name) if c.isalnum() and c.isalpha()]
    return (letters[0], letters[1]) if len(letters) >= 2 else None


def point_map(obj):
    result = {}
    for name, xy in (obj or {}).items():
        try:
            if len(xy) == 2:
                result[str(name)] = (float(xy[0]), float(xy[1]))
        except (TypeError, ValueError):
            continue
    return result


def rasterize(ex_path: Path, out_image: Path, out_mask: Path):
    try:
        ex = json.loads(ex_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"id": ex_path.parent.name, "status": "invalid_json", "error": str(exc)}

    width, height = int(ex.get("img_width", 0)), int(ex.get("img_height", 0))
    logic = ex.get("logic_form") or {}
    points = point_map(logic.get("point_positions"))
    if width <= 0 or height <= 0 or not points:
        return {"id": ex.get("id", ex_path.parent.name), "status": "missing_geometry"}

    image_path = ex_path.parent / "image.png"
    if not image_path.exists():
        return {"id": ex.get("id", ex_path.parent.name), "status": "missing_image"}

    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    line_count = 0
    missing_lines = []
    for raw in logic.get("line_instances") or []:
        pair = parse_line(raw)
        if not pair or pair[0] not in points or pair[1] not in points:
            missing_lines.append(str(raw))
            continue
        draw.line([points[pair[0]], points[pair[1]]], fill=1, width=max(2, round(min(width, height) / 120)))
        line_count += 1

    circle_count = 0
    for center_name in logic.get("circle_instances") or []:
        center = points.get(str(center_name))
        if center is None:
            continue
        radii = [math.hypot(x - center[0], y - center[1]) for name, (x, y) in points.items() if name != str(center_name)]
        if not radii:
            continue
        radius = max(radii)
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(bbox, outline=3, width=max(2, round(min(width, height) / 120)))
        circle_count += 1

    point_radius = max(2, round(min(width, height) / 75))
    for x, y in points.values():
        draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=4)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image)
    mask.save(out_mask)
    return {
        "id": ex.get("id", ex_path.parent.name),
        "status": "ok",
        "width": width,
        "height": height,
        "points": len(points),
        "lines": line_count,
        "circles": circle_count,
        "missing_lines": missing_lines,
        "source": str(ex_path.parent),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for ex_path in sorted(args.raw.rglob("ex.json")):
        sample_id = ex_path.parent.name
        records.append(rasterize(ex_path, args.out / "images" / f"{sample_id}.png", args.out / "masks" / f"{sample_id}.png"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    ok = sum(r.get("status") == "ok" for r in records)
    print(f"processed={len(records)} ok={ok} skipped={len(records) - ok}")


if __name__ == "__main__":
    main()
