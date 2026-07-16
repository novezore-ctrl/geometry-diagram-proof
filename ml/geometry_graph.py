"""Deterministic conversion from primitive candidates to GPT-friendly JSON."""
from __future__ import annotations

import json
from typing import Any


def to_graph(
    points: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    circles: list[dict[str, Any]],
    ellipses: list[dict[str, Any]] | None = None,
    labels: list[dict[str, Any]] | None = None,
    arrows: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "geometry-diagram/v2",
        "points": points,
        "segments": segments,
        "arrows": arrows or [],
        "attachments": attachments or [],
        "circles": circles,
        "ellipses": ellipses or [],
        "labels": labels or [],
        "relations": [],
        "uncertain": [
            "Coordinates are relative visual estimates unless the user confirms an exact relation.",
            "An arrow tip represents ray direction and is not an ordinary point.",
        ],
    }


def to_prompt_json(graph: dict[str, Any]) -> str:
    return json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=False)
