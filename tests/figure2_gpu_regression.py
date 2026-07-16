"""GPU acceptance test for the user's photographed Figure 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.gpu_runtime import GPUInferenceRuntime  # noqa: E402


def pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a locally owned/authorized Figure 2 photo (not redistributed by this repository).",
    )
    args = parser.parse_args()
    root = ROOT
    expected = json.loads((root / "tests/fixtures/figure2_expected.json").read_text(encoding="utf-8"))
    with Image.open(Path(args.image).expanduser().resolve()) as source:
        image = source.convert("RGB")
    runtime = GPUInferenceRuntime(args.checkpoint, memory_fraction=0.25)
    graph = runtime.infer(image, expected["questionText"])

    actual_labels = {point["label"] for point in graph["points"]}
    actual_segments = {pair(segment["a"], segment["b"]) for segment in graph["segments"]}
    expected_segments = {pair(first, second) for first, second in expected["segments"]}
    actual_attachments = {
        (item["junction"], pair(item["hostA"], item["hostB"])) for item in graph["attachments"]
    }
    expected_attachments = {
        (item["junction"], pair(*item["host"])) for item in expected["attachments"]
    }
    assert actual_labels == set(expected["points"]), (actual_labels, expected["points"])
    assert actual_segments == expected_segments, (actual_segments, expected_segments)
    assert len(graph["arrows"]) == expected["arrows"], graph["arrows"]
    assert actual_attachments == expected_attachments, (actual_attachments, expected_attachments)
    assert graph["metadata"]["device"] == "cuda"
    assert graph["metadata"]["batch_size"] == 1
    print(json.dumps({
        "accepted": True,
        "points": len(graph["points"]),
        "segments": len(graph["segments"]),
        "arrows": len(graph["arrows"]),
        "attachments": len(graph["attachments"]),
        "device": graph["metadata"]["device_name"],
        "batchSize": graph["metadata"]["batch_size"],
        "meanLineSupport": graph["metadata"]["diagnostics"]["mean_required_line_support"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
