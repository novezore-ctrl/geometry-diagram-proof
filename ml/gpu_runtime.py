"""Single-image CUDA runtime used by the local web inference service."""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from .extract_graph import extract
    from .model import CLASSES, GeometryUNet
    from .question_constraints import extract_question_constrained
except ImportError:
    from extract_graph import extract
    from model import CLASSES, GeometryUNet
    from question_constraints import extract_question_constrained


def image_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def scale_graph(graph: dict[str, Any], source_size: tuple[int, int], target_size: tuple[int, int]) -> dict[str, Any]:
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


class GPUInferenceRuntime:
    """Load one checkpoint once and serialize batch-1 CUDA inference."""

    def __init__(self, checkpoint_path: str | Path, memory_fraction: float = 0.25):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
        self.device = torch.device("cuda")
        self.batch_size = 1
        self.memory_fraction = float(memory_fraction)
        torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.size = tuple(int(value) for value in checkpoint.get("size", (320, 320)))
        self.epoch = int(checkpoint.get("epoch", 0))
        self.classes = list(checkpoint.get("classes", CLASSES))
        self.model = GeometryUNet().to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"])
        self.lock = Lock()

    def health(self) -> dict[str, Any]:
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "ready": True,
            "device": "cuda",
            "deviceName": torch.cuda.get_device_name(0),
            "dedicatedMemoryMiB": round(properties.total_memory / 1024 ** 2),
            "cudaFreeMiB": round(free_bytes / 1024 ** 2),
            "cudaTotalMiB": round(total_bytes / 1024 ** 2),
            "batchSize": self.batch_size,
            "memoryFractionLimit": self.memory_fraction,
            "checkpoint": str(self.checkpoint_path),
            "checkpointEpoch": self.epoch,
            "modelInput": list(self.size),
        }

    def infer(self, image: Image.Image, question_text: str = "") -> dict[str, Any]:
        image = image.convert("RGB")
        original_size = image.size
        tensor = image_tensor(image, self.size).to(self.device)
        with self.lock, torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            logits = self.model(tensor)
            probabilities = logits.softmax(dim=1)
            mask = probabilities.argmax(dim=1)[0].byte().cpu().numpy()
            mean_confidence = float(probabilities.max(dim=1).values.mean().item())
        graph = extract_question_constrained(mask, question_text)
        constrained = graph is not None
        if graph is None:
            graph = extract(mask)
        graph = scale_graph(graph, self.size, original_size)
        metadata = graph.setdefault("metadata", {})
        metadata.update({
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": self.epoch,
            "classes": self.classes,
            "model_size": list(self.size),
            "original_size": list(original_size),
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "batch_size": self.batch_size,
            "amp": True,
            "mean_pixel_confidence": round(mean_confidence, 4),
            "question_constraints_applied": constrained,
        })
        return graph

