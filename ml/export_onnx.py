"""Export the trained segmentation U-Net for on-device browser inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

try:
    from .model import GeometryUNet
except ImportError:
    from model import GeometryUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    height, width = (int(value) for value in state.get("size", (320, 320)))
    model = GeometryUNet().eval()
    model.load_state_dict(state["model"])
    example = torch.zeros(1, 3, height, width, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The legacy exporter is intentional here: it produces a compact static
    # graph without requiring the heavier onnxscript toolchain.
    torch.onnx.export(
        model,
        example,
        str(args.out),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )
    exported = onnx.load(args.out)
    onnx.checker.check_model(exported)
    inputs = [item.name for item in exported.graph.input]
    outputs = [item.name for item in exported.graph.output]
    print({
        "out": str(args.out),
        "bytes": args.out.stat().st_size,
        "opset": args.opset,
        "input": inputs,
        "output": outputs,
        "shape": [1, 3, height, width],
        "classes": list(state.get("classes", [])),
        "checkpoint_epoch": int(state.get("epoch", 0)),
    })


if __name__ == "__main__":
    main()
