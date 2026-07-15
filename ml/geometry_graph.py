"""Deterministic conversion from primitive candidates to GPT-friendly JSON."""
from __future__ import annotations

import json
from typing import Any


def to_graph(points: list[dict[str, Any]], segments: list[dict[str, Any]], circles: list[dict[str, Any]], ellipses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"schema": "geometry-diagram/v1", "points": points, "segments": segments, "circles": circles, "ellipses": ellipses or [], "relations": [], "uncertain": []}


def to_prompt_json(graph: dict[str, Any]) -> str:
    return json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=False)
