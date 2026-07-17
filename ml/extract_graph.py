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

try:  # package import in tests/server
    from .geometry_graph import to_graph, to_prompt_json
except ImportError:  # direct script execution
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


def component_box(pixels):
    ys = np.array([p[0] for p in pixels], dtype=float)
    xs = np.array([p[1] for p in pixels], dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def curve_candidates(mask: np.ndarray):
    circles, ellipses = [], []
    for pixels in components(mask == 3, min_area=18):
        x1, y1, x2, y2 = component_box(pixels)
        width, height = x2 - x1 + 1, y2 - y1 + 1
        if width < 8 or height < 8:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = width / 2, height / 2
        confidence = round(min(1.0, len(pixels) / max(24.0, np.pi * (rx + ry))), 3)
        if 0.85 <= rx / max(1.0, ry) <= 1.15:
            circles.append({
                "id": f"C{len(circles) + 1}", "cx": round(cx, 2), "cy": round(cy, 2),
                "r": round((rx + ry) / 2, 2), "confidence": confidence,
            })
        else:
            ellipses.append({
                "id": f"E{len(ellipses) + 1}", "cx": round(cx, 2), "cy": round(cy, 2),
                "rx": round(rx, 2), "ry": round(ry, 2), "confidence": confidence,
            })
    return circles, ellipses


def label_candidates(mask: np.ndarray):
    labels = []
    for pixels in components(mask == 5, min_area=4):
        x1, y1, x2, y2 = component_box(pixels)
        width, height = x2 - x1 + 1, y2 - y1 + 1
        if width > 80 or height > 80:
            continue
        labels.append({
            "id": f"L{len(labels) + 1}", "x": round(x1, 2), "y": round(y1, 2),
            "w": round(width, 2), "h": round(height, 2), "text": None,
            "confidence": round(min(1.0, len(pixels) / 30), 3),
        })
    return labels


def point_and_arrow_candidates(mask: np.ndarray, min_point_area: int = 5):
    """Split point-class blobs into ordinary points and one-sided arrow tips.

    The six-class model has no dedicated arrow class. Arrowheads are therefore
    recovered conservatively: an arrow blob must have exactly one dominant
    shaft direction in the surrounding line mask. Dots on a line have two
    opposite directions, while polygon vertices normally have two or more.
    """
    line_mask = (mask == 1) | (mask == 2)
    line_y, line_x = np.where(line_mask)
    points, arrows = [], []
    for pixels in components(mask == 4, min_point_area):
        ys = np.array([p[0] for p in pixels], dtype=float)
        xs = np.array([p[1] for p in pixels], dtype=float)
        cx, cy = float(xs.mean()), float(ys.mean())
        width, height = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        inner = max(4.5, max(width, height) * 0.62)
        radius = np.hypot(line_x - cx, line_y - cy)
        nearby = (radius >= inner) & (radius <= inner + 22)
        angles = (np.arctan2(line_y[nearby] - cy, line_x[nearby] - cx) + 2 * np.pi) % (2 * np.pi)
        histogram, _ = np.histogram(angles, bins=32, range=(0, 2 * np.pi))
        smooth = histogram + np.roll(histogram, 1) + np.roll(histogram, -1)
        peak_order = np.argsort(smooth)[::-1]
        peaks: list[int] = []
        maximum = int(smooth[peak_order[0]]) if peak_order.size else 0
        for raw_index in peak_order:
            index = int(raw_index)
            circular_distance = lambda other: min(abs(index - other), 32 - abs(index - other))
            if smooth[index] < max(3, maximum * 0.34):
                break
            if all(circular_distance(other) >= 3 for other in peaks):
                peaks.append(index)
            if len(peaks) == 3:
                break

        one_sided = len(peaks) == 1 and maximum >= 5 and len(pixels) >= 12
        if one_sided:
            inward_angle = (peaks[0] + 0.5) * 2 * np.pi / 32
            direction_x, direction_y = -float(np.cos(inward_angle)), -float(np.sin(inward_angle))
            projection = (xs - cx) * direction_x + (ys - cy) * direction_y
            tip_at = int(np.argmax(projection))
            arrows.append({
                "id": f"A{len(arrows) + 1}",
                "tip": {"x": round(float(xs[tip_at]), 2), "y": round(float(ys[tip_at]), 2)},
                "direction": {"x": round(direction_x, 4), "y": round(direction_y, 4)},
                "from": None,
                "confidence": round(min(0.98, 0.55 + maximum / max(20.0, float(histogram.sum())) * 0.35), 3),
                "source": "mask_postprocess",
            })
        else:
            points.append({
                "id": f"P{len(points) + 1}", "x": round(cx, 2), "y": round(cy, 2),
                "confidence": round(min(1.0, len(pixels) / 40), 3),
            })
    return points, arrows


def extract(mask: np.ndarray, min_point_area=5, support_threshold=0.58):
    points, arrows = point_and_arrow_candidates(mask, min_point_area)
    line_mask = (mask == 1) | (mask == 2)
    segments = []
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            b = points[j]
            xs, ys = sample_segment((a["x"], a["y"]), (b["x"], b["y"]))
            valid = (xs >= 0) & (ys >= 0) & (ys < mask.shape[0]) & (xs < mask.shape[1])
            support = float(line_mask[ys[valid], xs[valid]].mean()) if valid.any() else 0.0
            if support >= support_threshold:
                line_values = mask[ys[valid], xs[valid]]
                line_values = line_values[(line_values == 1) | (line_values == 2)]
                dashed_fraction = float((line_values == 2).mean()) if line_values.size else 0.0
                style = "dashed" if dashed_fraction >= 0.5 else "mixed" if dashed_fraction >= 0.1 else "solid"
                segments.append({
                    "a": a["id"], "b": b["id"], "support": round(support, 3),
                    "style": style, "dashed_fraction": round(dashed_fraction, 3),
                })
    for arrow in arrows:
        tip = arrow["tip"]
        matches = []
        for point in points:
            distance = float(np.hypot(tip["x"] - point["x"], tip["y"] - point["y"]))
            if distance < 4:
                continue
            outward_x = (tip["x"] - point["x"]) / distance
            outward_y = (tip["y"] - point["y"]) / distance
            alignment = outward_x * arrow["direction"]["x"] + outward_y * arrow["direction"]["y"]
            if alignment < 0.72:
                continue
            xs, ys = sample_segment((tip["x"], tip["y"]), (point["x"], point["y"]))
            valid = (xs >= 0) & (ys >= 0) & (ys < mask.shape[0]) & (xs < mask.shape[1])
            support = float(line_mask[ys[valid], xs[valid]].mean()) if valid.any() else 0.0
            if support >= 0.48:
                matches.append((distance, point["id"], support))
        if matches:
            distance, point_id, support = min(matches)
            arrow["from"] = point_id
            arrow["shaft_support"] = round(support, 3)
    circles, ellipses = curve_candidates(mask)
    return to_graph(points, segments, circles, ellipses, label_candidates(mask), arrows=arrows, attachments=[])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mask", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph = extract(np.asarray(Image.open(args.mask)))
    args.out.write_text(to_prompt_json(graph) + "\n", encoding="utf-8")
    print(json.dumps({"points": len(graph["points"]), "segments": len(graph["segments"]), "arrows": len(graph["arrows"]), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
