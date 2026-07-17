"""Evaluate segmentation and point-connected topology on processed pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from extract_graph import extract
from model import CLASSES, GeometryUNet


class PairDataset(Dataset):
    def __init__(self, root: Path, split: str, size: tuple[int, int], limit: int | None):
        image_dir, mask_dir = root / "images", root / "masks"
        pairs = [(p, mask_dir / p.name) for p in sorted(image_dir.glob(f"{split}_*.png")) if (mask_dir / p.name).exists()]
        self.pairs = pairs[:limit] if limit else pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        height, width = self.size
        image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_array = np.array(mask, dtype=np.int64, copy=True)
        return torch.from_numpy(image_array), torch.from_numpy(mask_array), image_path.name


def match_topology(pred_graph: dict, true_graph: dict, tolerance: float):
    pred_points, true_points = pred_graph["points"], true_graph["points"]
    candidates = []
    for pred in pred_points:
        for true in true_points:
            distance = math.hypot(pred["x"] - true["x"], pred["y"] - true["y"])
            if distance <= tolerance:
                candidates.append((distance, pred["id"], true["id"]))
    matched_pred, matched_true, mapping = set(), set(), {}
    for _, pred_id, true_id in sorted(candidates):
        if pred_id not in matched_pred and true_id not in matched_true:
            matched_pred.add(pred_id); matched_true.add(true_id); mapping[pred_id] = true_id

    true_edges = {tuple(sorted((edge["a"], edge["b"]))) for edge in true_graph["segments"]}
    mapped_edges, unmatched_edges = set(), 0
    for edge in pred_graph["segments"]:
        if edge["a"] not in mapping or edge["b"] not in mapping:
            unmatched_edges += 1
            continue
        mapped_edges.add(tuple(sorted((mapping[edge["a"]], mapping[edge["b"]]))))
    edge_tp = len(mapped_edges & true_edges)
    return {
        "point_tp": len(mapping), "point_fp": len(pred_points) - len(mapping), "point_fn": len(true_points) - len(mapping),
        "edge_tp": edge_tp, "edge_fp": len(mapped_edges - true_edges) + unmatched_edges,
        "edge_fn": len(true_edges - mapped_edges),
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--point-tolerance", type=float, default=10.0)
    parser.add_argument("--support-threshold", type=float, default=0.58)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the project evaluation")
    if args.batch_size > 16:
        raise SystemExit("This machine is safety-capped at batch size 16 for this project")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    size = tuple(int(v) for v in checkpoint.get("size", (320, 320)))
    dataset = PairDataset(args.data, args.split, size, args.limit)
    if not dataset:
        raise SystemExit(f"No {args.split!r} pairs found under {args.data}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    model = GeometryUNet().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    class_count = len(CLASSES)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    topology = {key: 0 for key in ("point_tp", "point_fp", "point_fn", "edge_tp", "edge_fp", "edge_fn")}
    exact_graphs = 0
    failures = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for batch_index, (images, masks, names) in enumerate(loader, start=1):
            images_gpu = images.to(device, non_blocking=True)
            masks_gpu = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(images_gpu).argmax(dim=1)
            encoded = masks_gpu.reshape(-1) * class_count + pred.reshape(-1)
            confusion += torch.bincount(encoded, minlength=class_count**2).reshape(class_count, class_count)
            predicted_masks = pred.byte().cpu().numpy()
            true_masks = masks.numpy().astype(np.uint8, copy=False)
            for name, predicted_mask, true_mask in zip(names, predicted_masks, true_masks):
                predicted_graph = extract(predicted_mask, support_threshold=args.support_threshold)
                true_graph = extract(true_mask, support_threshold=args.support_threshold)
                sample = match_topology(predicted_graph, true_graph, args.point_tolerance)
                for key, value in sample.items(): topology[key] += value
                errors = sample["point_fp"] + sample["point_fn"] + sample["edge_fp"] + sample["edge_fn"]
                if errors == 0:
                    exact_graphs += 1
                else:
                    failures.append({"image": name, "errors": errors, **sample})
                    failures = sorted(failures, key=lambda item: item["errors"], reverse=True)[:40]
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"batch={batch_index}/{len(loader)} images={min(batch_index * args.batch_size, len(dataset))} elapsed_s={time.perf_counter() - started:.1f}", flush=True)

    confusion_cpu = confusion.cpu()
    intersection = confusion_cpu.diag()
    actual, predicted = confusion_cpu.sum(1), confusion_cpu.sum(0)
    union = actual + predicted - intersection
    class_metrics = {}
    valid_ious = []
    for index, name in enumerate(CLASSES):
        precision = ratio(int(intersection[index]), int(predicted[index]))
        recall = ratio(int(intersection[index]), int(actual[index]))
        iou = ratio(int(intersection[index]), int(union[index]))
        if int(union[index]): valid_ious.append(iou)
        class_metrics[name] = {"precision": round(precision, 6), "recall": round(recall, 6), "iou": round(iou, 6), "actual_pixels": int(actual[index])}

    point_tp, edge_tp = topology["point_tp"], topology["edge_tp"]
    elapsed = time.perf_counter() - started
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "data": str(args.data), "split": args.split, "images": len(dataset),
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "model_size": list(size), "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(elapsed, 3),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "pixel_accuracy": round(ratio(int(intersection.sum()), int(confusion_cpu.sum())), 6),
        "mean_iou_present_classes": round(sum(valid_ious) / len(valid_ious), 6),
        "classes": class_metrics,
        "topology": {
            "basis": "mask-derived proxy graph at 320x320; not direct scoring against original annotation relations",
            "point_tolerance_pixels": args.point_tolerance,
            "line_support_threshold": args.support_threshold,
            **topology,
            "point_precision": round(ratio(point_tp, point_tp + topology["point_fp"]), 6),
            "point_recall": round(ratio(point_tp, point_tp + topology["point_fn"]), 6),
            "edge_precision": round(ratio(edge_tp, edge_tp + topology["edge_fp"]), 6),
            "edge_recall": round(ratio(edge_tp, edge_tp + topology["edge_fn"]), 6),
            "exact_graph_rate": round(exact_graphs / len(dataset), 6),
        },
        "sample_failures": sorted(failures, key=lambda item: item["errors"], reverse=True),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "images": len(dataset), "mean_iou_present_classes": report["mean_iou_present_classes"], **report["topology"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
