"""Convert a predicted six-class mask into a point-connected geometry graph.

The key operation is pairwise line support: after detecting point blobs, a
segment is emitted only when the predicted line mask covers most samples
between two points. This preserves topology better than treating every line
pixel as an independent object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from geometry_graph import to_graph, to_prompt_json


def components(binary: np.ndarray, min_area: int = 4):
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    result = []
    for y, x in zip(*np.where(binary & ~seen)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(pixels) >= min_area:
            result.append(pixels)
    return result


def sample_segment(a, b, count=160):
    xs = np.linspace(a[0], b[0], count).round().astype(int)
    ys = np.linspace(a[1], b[1], count).round().astype(int)
    return xs, ys


def extract(mask: np.ndarray, min_point_area=5, support_threshold=0.58):
    point_components = components(mask == 4, min_point_area)
    points = []
    for idx, pixels in enumerate(point_components):
        ys = np.array([p[0] for p in pixels], dtype=float)
        xs = np.array([p[1] for p in pixels], dtype=float)
        points.append({"id": f"P{idx + 1}", "x": round(float(xs.mean()), 2), "y": round(float(ys.mean()), 2), "confidence": round(min(1.0, len(pixels) / 40), 3)})
    line_mask = (mask == 1) | (mask == 2)
    segments = []
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            b = points[j]
            xs, ys = sample_segment((a["x"], a["y"]), (b["x"], b["y"]))
            valid = (xs >= 0) & (ys >= 0) & (ys < mask.shape[0]) & (xs < mask.shape[1])
            support = float(line_mask[ys[valid], xs[valid]].mean()) if valid.any() else 0.0
            if support >= support_threshold:
                segments.append({"a": a["id"], "b": b["id"], "support": round(support, 3), "style": "mixed" if (mask[ys[valid], xs[valid]] == 2).any() else "solid"})
    return to_graph(points, segments, [], [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph = extract(np.asarray(Image.open(args.mask).convert("L")))
    args.out.write_text(to_prompt_json(graph) + "\n", encoding="utf-8")
    print(json.dumps({"points": len(graph["points"]), "segments": len(graph["segments"]), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
