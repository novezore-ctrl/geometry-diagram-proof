"""Question-aware topology recovery for photographed geometry diagrams.

The U-Net is a pixel segmenter, not a complete graph recognizer.  Printed
letters and angle marks can share the point class, so exporting every point
blob and every supported pair creates a combinatorial false graph.  This
module uses explicit relations from the problem statement as a graph template
and assigns only the best-supported U-Net point candidates to that template.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot
from typing import Any

import numpy as np

try:  # package import in tests/server
    from .extract_graph import point_and_arrow_candidates, sample_segment
    from .geometry_graph import to_graph
except ImportError:  # direct execution beside infer.py
    from extract_graph import point_and_arrow_candidates, sample_segment
    from geometry_graph import to_graph


@dataclass(frozen=True)
class TopologyTemplate:
    name: str
    labels: tuple[str, ...]
    segments: tuple[tuple[str, str], ...]
    ordered_chains: tuple[tuple[str, str, str], ...]


FIGURE2_TEMPLATE = TopologyTemplate(
    name="fold_angle_bisector_figure2",
    labels=("A", "B", "C", "D", "E", "F", "G"),
    segments=(
        ("E", "G"), ("B", "C"), ("B", "D"), ("D", "A"),
        ("C", "D"), ("D", "G"), ("C", "A"), ("E", "D"),
        ("B", "E"), ("E", "C"),
    ),
    ordered_chains=(
        ("A", "D", "B"), ("C", "D", "G"),
        ("B", "F", "C"), ("E", "F", "D"),
    ),
)


def template_from_question(question_text: str) -> TopologyTemplate | None:
    """Recognize the known Figure 2 construction from its explicit wording.

    This deliberately requires several independent phrases.  A partial or
    unrelated question must not silently receive this topology.
    """
    compact = "".join((question_text or "").upper().split())
    evidence = (
        "D" in compact and "AB" in compact,
        ("翻折" in compact or "折叠" in compact) and "CD" in compact,
        "DE" in compact and "CB" in compact and "F" in compact,
        "连接BE" in compact,
        "BED" in compact and "平分线" in compact and "G" in compact,
    )
    return FIGURE2_TEMPLATE if all(evidence) else None


def _merge_candidates(points: list[dict[str, Any]], radius: float = 7.5) -> list[dict[str, float]]:
    """Merge split blobs belonging to one physical printed vertex."""
    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, first in enumerate(points):
        for j in range(i + 1, len(points)):
            second = points[j]
            if hypot(first["x"] - second["x"], first["y"] - second["y"]) <= radius:
                union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, point in enumerate(points):
        groups.setdefault(find(index), []).append(point)

    merged = []
    for members in groups.values():
        weights = [max(0.15, float(item.get("confidence", 0.5))) for item in members]
        total = sum(weights)
        merged.append({
            "x": sum(float(item["x"]) * weight for item, weight in zip(members, weights)) / total,
            "y": sum(float(item["y"]) * weight for item, weight in zip(members, weights)) / total,
            "confidence": max(float(item.get("confidence", 0.5)) for item in members),
            "members": float(len(members)),
        })
    return merged


def _dilate(binary: np.ndarray, radius: int = 2) -> np.ndarray:
    output = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            source_y1, source_y2 = max(0, -dy), min(height, height - dy)
            source_x1, source_x2 = max(0, -dx), min(width, width - dx)
            target_y1, target_y2 = source_y1 + dy, source_y2 + dy
            target_x1, target_x2 = source_x1 + dx, source_x2 + dx
            output[target_y1:target_y2, target_x1:target_x2] |= binary[source_y1:source_y2, source_x1:source_x2]
    return output


def _line_support(line_mask: np.ndarray, first: dict[str, float], second: dict[str, float]) -> float:
    xs, ys = sample_segment((first["x"], first["y"]), (second["x"], second["y"]), count=220)
    valid = (xs >= 0) & (ys >= 0) & (ys < line_mask.shape[0]) & (xs < line_mask.shape[1])
    if not valid.any():
        return 0.0
    return float(line_mask[ys[valid], xs[valid]].mean())


def _ordered_collinearity(first: dict[str, float], middle: dict[str, float], last: dict[str, float]) -> float:
    dx, dy = last["x"] - first["x"], last["y"] - first["y"]
    length2 = dx * dx + dy * dy
    if length2 < 36:
        return 0.0
    t = ((middle["x"] - first["x"]) * dx + (middle["y"] - first["y"]) * dy) / length2
    projected_x, projected_y = first["x"] + t * dx, first["y"] + t * dy
    offset = hypot(middle["x"] - projected_x, middle["y"] - projected_y)
    length = length2 ** 0.5
    tolerance = max(3.5, min(10.0, length * 0.045))
    alignment = exp(-((offset / tolerance) ** 2))
    between = 1.0 if 0.04 < t < 0.96 else max(0.0, 1.0 - abs(t - 0.5) * 2.0)
    return alignment * between


def _assignment_score(
    assignment: dict[str, int],
    candidates: list[dict[str, float]],
    template: TopologyTemplate,
    support: dict[tuple[int, int], float],
) -> float:
    score = 0.0
    for label, index in assignment.items():
        candidate = candidates[index]
        score += 0.07 * candidate["confidence"] + 0.025 * min(2.0, candidate["members"])
    for first_label, second_label in template.segments:
        if first_label in assignment and second_label in assignment:
            pair = tuple(sorted((assignment[first_label], assignment[second_label])))
            value = support[pair]
            # Supported lines are useful, but a weak/absent required line must
            # also be expensive; otherwise angle marks can win by chance.
            score += 2.8 * value - 1.1 * max(0.0, 0.52 - value)
    for first_label, middle_label, last_label in template.ordered_chains:
        if all(label in assignment for label in (first_label, middle_label, last_label)):
            value = _ordered_collinearity(
                candidates[assignment[first_label]],
                candidates[assignment[middle_label]],
                candidates[assignment[last_label]],
            )
            score += 5.2 * value - 2.0 * (1.0 - value)
    return score


def _solve_template(mask: np.ndarray, template: TopologyTemplate) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    raw_points, _raw_arrows = point_and_arrow_candidates(mask)
    candidates = _merge_candidates(raw_points)
    height, width = mask.shape
    margin = max(3.0, min(width, height) * 0.018)
    candidates = [
        item for item in candidates
        if margin <= item["x"] <= width - margin and margin <= item["y"] <= height - margin
    ]
    if len(candidates) < len(template.labels):
        raise ValueError(f"only {len(candidates)} point candidates for {len(template.labels)} constrained points")

    line_mask = _dilate((mask == 1) | (mask == 2), radius=2)
    support: dict[tuple[int, int], float] = {}
    for i, first in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            support[(i, j)] = _line_support(line_mask, first, candidates[j])

    # Add roles in an order that exposes constraints early.  Beam search keeps
    # the solver deterministic while avoiding an exhaustive N-permutation-7.
    role_order = ("D", "F", "A", "B", "C", "G", "E")
    beam: list[tuple[float, dict[str, int]]] = [(0.0, {})]
    beam_width = 2400
    for role in role_order:
        expanded: list[tuple[float, dict[str, int]]] = []
        for _old_score, assignment in beam:
            used = set(assignment.values())
            for index in range(len(candidates)):
                if index in used:
                    continue
                next_assignment = {**assignment, role: index}
                expanded.append((_assignment_score(next_assignment, candidates, template, support), next_assignment))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beam = expanded[:beam_width]
    if not beam:
        raise ValueError("question-constrained point assignment failed")

    best_score, best = beam[0]
    assigned = {label: candidates[index] for label, index in best.items()}
    edge_support = {
        f"{first}-{second}": support[tuple(sorted((best[first], best[second])))]
        for first, second in template.segments
    }
    diagnostics = {
        "assignment_score": round(best_score, 4),
        "candidate_count_raw": float(len(raw_points)),
        "candidate_count_merged": float(len(candidates)),
        "mean_required_line_support": round(sum(edge_support.values()) / len(edge_support), 4),
        **{f"support_{key}": round(value, 4) for key, value in edge_support.items()},
    }
    return assigned, diagnostics


def extract_question_constrained(mask: np.ndarray, question_text: str) -> dict[str, Any] | None:
    """Return a constrained graph, or ``None`` when the text is insufficient."""
    template = template_from_question(question_text)
    if template is None:
        return None
    assigned, diagnostics = _solve_template(mask, template)
    points = [
        {
            "id": label,
            "label": label,
            "x": round(assigned[label]["x"], 2),
            "y": round(assigned[label]["y"], 2),
            "confidence": round(max(0.55, min(0.98, assigned[label]["confidence"])), 3),
            "source": "question_constrained",
            "label_source": "question_topology",
        }
        for label in template.labels
    ]
    segments = []
    for index, (first, second) in enumerate(template.segments, start=1):
        segments.append({
            "id": f"S{index}", "a": first, "b": second,
            "confidence": round(diagnostics[f"support_{first}-{second}"], 3),
            "source": "question_constrained", "style": "solid",
        })
    attachments = []
    for index, (first, middle, last) in enumerate(template.ordered_chains, start=1):
        first_point, middle_point, last_point = assigned[first], assigned[middle], assigned[last]
        dx, dy = last_point["x"] - first_point["x"], last_point["y"] - first_point["y"]
        length2 = max(1e-9, dx * dx + dy * dy)
        t = ((middle_point["x"] - first_point["x"]) * dx + (middle_point["y"] - first_point["y"]) * dy) / length2
        attachments.append({
            "id": f"O{index}", "kind": "point_on_segment", "junction": middle,
            "hostA": first, "hostB": last,
            "position": {"kind": "approximate", "approximateT": round(max(0.0, min(1.0, t)), 4), "coordinateUncertain": True},
            "confidence": round(max(0.55, _ordered_collinearity(first_point, middle_point, last_point)), 3),
            "source": "question_constrained", "support": "question_and_collinearity",
        })
    graph = to_graph(points, segments, [], [], [], arrows=[], attachments=attachments)
    graph["metadata"] = {
        "extraction_mode": "question_constrained",
        "template": template.name,
        "label_policy": "names assigned from question topology; glyph OCR not claimed",
        "expected_counts": {"points": 7, "segments": 10, "arrows": 0, "attachments": 4},
        "diagnostics": diagnostics,
    }
    return graph
