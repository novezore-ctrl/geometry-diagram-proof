"""Export a trained checkpoint to Core ML when coremltools is installed."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from model import GeometryUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Core ML .mlpackage output")
    parser.add_argument("--trace-out", type=Path, default=None, help="optional verified TorchScript intermediate")
    parser.add_argument("--trace-only", action="store_true", help="stop after TorchScript export; coremltools is not required")
    args = parser.parse_args()
    model = GeometryUNet().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    height, width = (int(v) for v in state.get("size", (320, 320)))
    example = torch.rand(1, 3, height, width)
    traced = torch.jit.trace(model, example)
    with torch.inference_mode():
        difference = float((model(example) - traced(example)).abs().max())
    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(args.trace_out))
        reloaded = torch.jit.load(str(args.trace_out)).eval()
        with torch.inference_mode():
            output = reloaded(example)
        if tuple(output.shape) != (1, len(state["classes"]), height, width) or not torch.isfinite(output).all():
            raise SystemExit("TorchScript verification failed")
        print(f"saved {args.trace_out} input={width}x{height} max_trace_difference={difference:.8f}")
    if args.trace_only:
        if not args.trace_out:
            raise SystemExit("--trace-only requires --trace-out")
        return
    if not args.out:
        raise SystemExit("--out is required unless --trace-only is used")
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit("Install coremltools on macOS or a compatible conversion environment first.") from exc
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, height, width), scale=1 / 255.0, bias=[0, 0, 0])],
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.author = "Geometry Diagram Parser"
    mlmodel.short_description = "Six-class geometry diagram segmentation"
    mlmodel.input_description["image"] = f"RGB geometry crop resized to {width}x{height}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print(f"saved {args.out} input={width}x{height}")


if __name__ == "__main__": main()
