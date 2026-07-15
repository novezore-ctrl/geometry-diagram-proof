"""Export a trained checkpoint to Core ML when coremltools is installed."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from model import GeometryUNet


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit("Install coremltools on macOS or a compatible conversion environment first.") from exc
    model = GeometryUNet().eval(); state = torch.load(args.checkpoint, map_location="cpu"); model.load_state_dict(state["model"])
    traced = torch.jit.trace(model, torch.rand(1, 3, 256, 256)); mlmodel = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=(1, 3, 256, 256), scale=1 / 255.0, bias=[0, 0, 0])], minimum_deployment_target=ct.target.iOS16)
    args.out.parent.mkdir(parents=True, exist_ok=True); mlmodel.save(str(args.out)); print(f"saved {args.out}")


if __name__ == "__main__": main()
