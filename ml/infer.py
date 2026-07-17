"""Run a trained GeometryUNet checkpoint and export mask plus graph JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from extract_graph import extract
from geometry_graph import to_prompt_json
from model import CLASSES, GeometryUNet


PALETTE = [
    255, 255, 255, 35, 105, 210, 255, 140, 0, 145, 85, 210,
    15, 155, 105, 230, 85, 70,
] + [0] * (256 * 3 - 18)


def image_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def scale_graph(graph: dict, source_size: tuple[int, int], target_size: tuple[int, int]):
    source_h, source_w = source_size
    target_w, target_h = target_size
    sx, sy = target_w / source_w, target_h / source_h
    for point in graph["points"]:
        point["x"], point["y"] = round(point["x"] * sx, 2), round(point["y"] * sy, 2)
    for arrow in graph["arrows"]:
        arrow["tip"]["x"], arrow["tip"]["y"] = round(arrow["tip"]["x"] * sx, 2), round(arrow["tip"]["y"] * sy, 2)
        dx, dy = arrow["direction"]["x"] * sx, arrow["direction"]["y"] * sy
        length = max(1e-9, float(np.hypot(dx, dy)))
        arrow["direction"]["x"], arrow["direction"]["y"] = round(dx / length, 4), round(dy / length, 4)
    for circle in graph["circles"]:
        circle["cx"], circle["cy"] = round(circle["cx"] * sx, 2), round(circle["cy"] * sy, 2)
        circle["r"] = round(circle["r"] * (sx + sy) / 2, 2)
    for ellipse in graph["ellipses"]:
        ellipse["cx"], ellipse["cy"] = round(ellipse["cx"] * sx, 2), round(ellipse["cy"] * sy, 2)
        ellipse["rx"], ellipse["ry"] = round(ellipse["rx"] * sx, 2), round(ellipse["ry"] * sy, 2)
    for label in graph["labels"]:
        label["x"], label["y"] = round(label["x"] * sx, 2), round(label["y"] * sy, 2)
        label["w"], label["h"] = round(label["w"] * sx, 2), round(label["h"] * sy, 2)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-mask", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.75)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    size = tuple(int(v) for v in checkpoint.get("size", (320, 320)))
    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    model = GeometryUNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    tensor = image_tensor(image, size).to(device)
    use_amp = bool(args.amp and device.type == "cuda")
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
        logits = model(tensor)
        probabilities = logits.softmax(dim=1)
        mask = probabilities.argmax(dim=1)[0].byte().cpu().numpy()
        mean_confidence = float(probabilities.max(dim=1).values.mean().item())

    graph = scale_graph(extract(mask), size, original_size)
    graph["metadata"] = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "classes": list(checkpoint.get("classes", CLASSES)),
        "model_size": list(size),
        "original_size": list(original_size),
        "device": str(device),
        "amp": use_amp,
        "mean_pixel_confidence": round(mean_confidence, 4),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(to_prompt_json(graph) + "\n", encoding="utf-8")

    if args.out_mask:
        args.out_mask.parent.mkdir(parents=True, exist_ok=True)
        mask_image = Image.fromarray(mask, mode="P")
        mask_image.putpalette(PALETTE)
        mask_image.resize(original_size, Image.Resampling.NEAREST).save(args.out_mask)

    print(json.dumps({
        "out_json": str(args.out_json), "out_mask": str(args.out_mask) if args.out_mask else None,
        "points": len(graph["points"]), "segments": len(graph["segments"]),
        "arrows": len(graph["arrows"]), "attachments": len(graph["attachments"]),
        "circles": len(graph["circles"]), "ellipses": len(graph["ellipses"]),
        "labels": len(graph["labels"]), "device": str(device), "amp": use_amp,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
