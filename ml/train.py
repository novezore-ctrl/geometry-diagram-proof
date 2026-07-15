"""Train GeometryUNet on synthetic pairs, optionally mixed with Geometry3K."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from model import CLASSES, GeometryUNet


SIZE = (320, 320)


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = TF.resize(image, SIZE, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, SIZE, interpolation=InterpolationMode.NEAREST)
        return TF.to_tensor(image), torch.from_numpy(np.asarray(mask, dtype=np.int64))


def synthetic_pairs(root: Path) -> list[tuple[Path, Path]]:
    return [(p, p.with_name(p.stem + ".mask.png")) for p in sorted(root.glob("*.png")) if not p.name.endswith(".mask.png") and p.with_name(p.stem + ".mask.png").exists()]


def real_pairs(root: Path) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = root / "images", root / "masks"
    return [(p, mask_dir / p.name) for p in sorted(image_dir.glob("*.png")) if (mask_dir / p.name).exists()]


def split_pairs(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    n_val = max(1, round(len(pairs) * val_ratio)) if pairs else 0
    val_ids, train_ids = order[:n_val], order[n_val:]
    return [pairs[i] for i in train_ids], [pairs[i] for i in val_ids]


def metrics(model, loader, device):
    model.eval()
    correct = total_pixels = 0
    intersection = torch.zeros(len(CLASSES), dtype=torch.long)
    union = torch.zeros(len(CLASSES), dtype=torch.long)
    predicted = torch.zeros(len(CLASSES), dtype=torch.long)
    actual = torch.zeros(len(CLASSES), dtype=torch.long)
    with torch.no_grad():
        for images, masks in loader:
            pred = model(images.to(device)).argmax(1).cpu()
            correct += int((pred == masks).sum())
            total_pixels += masks.numel()
            for cls in range(len(CLASSES)):
                p, m = pred == cls, masks == cls
                intersection[cls] += int((p & m).sum())
                union[cls] += int((p | m).sum())
                predicted[cls] += int(p.sum())
                actual[cls] += int(m.sum())
    valid = union > 0
    iou = intersection[valid].float() / union[valid].float().clamp_min(1)
    precision = intersection.float() / predicted.float().clamp_min(1)
    recall = intersection.float() / actual.float().clamp_min(1)
    return correct / max(1, total_pixels), iou.mean().item(), iou.tolist(), precision.tolist(), recall.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="synthetic image/.mask.png directory")
    parser.add_argument("--real-data", type=Path, default=None, help="Geometry3K processed directory")
    parser.add_argument("--real-repeat", type=int, default=2, help="repeat scarce real samples in the training mix")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    syn_train, syn_val = split_pairs(synthetic_pairs(args.data), 0.2, 42)
    train_sets = [PairDataset(syn_train)]
    val_sets = [PairDataset(syn_val)]
    real_count = 0
    if args.real_data:
        real_train, real_val = split_pairs(real_pairs(args.real_data), 0.2, 43)
        real_count = len(real_train) + len(real_val)
        if real_train:
            real_set = PairDataset(real_train)
            train_sets.extend([real_set] * max(1, args.real_repeat))
        if real_val:
            val_sets.append(PairDataset(real_val))

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = GeometryUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weights = torch.tensor([0.25, 1.15, 1.35, 1.25, 1.5, 1.3], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"device={device} synthetic_train={len(syn_train)} real_total={real_count} mixed_train={len(train_ds)} val={len(val_ds)}")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
        acc, mean_iou, ious, precision, recall = metrics(model, val_loader, device)
        # Class ids: 1=solid line, 3=curve, 4=point. These are more useful
        # than whole-image accuracy after the user has cropped a region.
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(loader)):.4f} "
            f"val_pixel_acc={acc:.4f} val_mean_iou={mean_iou:.4f} "
            f"line_p={precision[1]:.3f} line_r={recall[1]:.3f} "
            f"point_p={precision[4]:.3f} point_r={recall[4]:.3f} "
            f"curve_p={precision[3]:.3f} curve_r={recall[3]:.3f}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": CLASSES, "size": SIZE}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
