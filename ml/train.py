"""Train GeometryUNet on synthetic pairs, optionally mixed with Geometry3K."""
from __future__ import annotations

import argparse
import time
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


def real_pairs(root: Path, split: str | None = None) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = root / "images", root / "masks"
    pairs = [(p, mask_dir / p.name) for p in sorted(image_dir.glob("*.png")) if (mask_dir / p.name).exists()]
    if split:
        pairs = [(p, m) for p, m in pairs if p.stem.startswith(split + "_")]
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    n_val = max(1, round(len(pairs) * val_ratio)) if pairs else 0
    val_ids, train_ids = order[:n_val], order[n_val:]
    return [pairs[i] for i in train_ids], [pairs[i] for i in val_ids]


def metrics(model, loader, device, use_amp: bool):
    model.eval()
    class_count = len(CLASSES)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long, device=device)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                pred = model(images).argmax(1)
            encoded = masks.reshape(-1) * class_count + pred.reshape(-1)
            confusion += torch.bincount(encoded, minlength=class_count**2).reshape(class_count, class_count)
    intersection = confusion.diag()
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = actual + predicted - intersection
    valid = union > 0
    iou = intersection[valid].float() / union[valid].float().clamp_min(1)
    precision = intersection.float() / predicted.float().clamp_min(1)
    recall = intersection.float() / actual.float().clamp_min(1)
    accuracy = intersection.sum().float() / confusion.sum().float().clamp_min(1)
    return accuracy.item(), iou.mean().item(), iou.tolist(), precision.tolist(), recall.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="synthetic image/.mask.png directory")
    parser.add_argument("--real-data", type=Path, default=None, help="Geometry3K processed directory")
    parser.add_argument("--real-repeat", type=int, default=2, help="repeat scarce real samples in the training mix")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="use CUDA float16 mixed precision")
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of silently training on CPU")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.75,
        help="cap this process's CUDA allocator to prevent WDDM memory spill",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise SystemExit("CUDA was required, but torch.cuda.is_available() is false")
    use_amp = bool(args.amp and device == "cuda")
    dedicated_vram = 0
    if device == "cuda":
        if not 0 < args.cuda_memory_fraction <= 1:
            raise SystemExit("--cuda-memory-fraction must be in (0, 1]")
        dedicated_vram = torch.cuda.get_device_properties(0).total_memory
        if dedicated_vram <= 9 * 1024**3 and args.batch_size > 16:
            raise SystemExit(
                f"batch size {args.batch_size} exceeds this <=9 GiB GPU's safety limit of 16"
            )
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.backends.cudnn.benchmark = True

    syn_train, syn_val = split_pairs(synthetic_pairs(args.data), 0.2, 42)
    train_sets = [PairDataset(syn_train)]
    val_sets = [PairDataset(syn_val)]
    real_count = 0
    if args.real_data:
        named_train = real_pairs(args.real_data, "train")
        named_val = real_pairs(args.real_data, "val")
        if named_train and named_val:
            real_train, real_val = named_train, named_val
        else:
            real_train, real_val = split_pairs(real_pairs(args.real_data), 0.2, 43)
        real_count = len(real_train) + len(real_val)
        if real_train:
            real_set = PairDataset(real_train)
            train_sets.extend([real_set] * max(1, args.real_repeat))
        if real_val:
            val_sets.append(PairDataset(real_val))

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    model = GeometryUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
    weights = torch.tensor([0.25, 1.15, 1.35, 1.25, 1.5, 1.3], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    vram_summary = f"{dedicated_vram / 1024**3:.2f}" if device == "cuda" else "n/a"
    run_summary = (
        f"device={device} gpu={gpu_name!r} amp={use_amp} "
        f"dedicated_vram_gib={vram_summary} "
        f"cuda_memory_fraction={args.cuda_memory_fraction if device == 'cuda' else 'n/a'} "
        f"synthetic_train={len(syn_train)} real_total={real_count} "
        f"mixed_train={len(train_ds)} val={len(val_ds)} start_epoch={start_epoch}"
    )
    print(run_summary, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total = 0.0
        for step, (images, masks) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().item()
            if args.log_every > 0 and (step % args.log_every == 0 or step == len(loader)):
                peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {step}/{len(loader)} "
                    f"loss={total / step:.4f} elapsed_s={time.perf_counter() - epoch_started:.1f} "
                    f"peak_vram_gib={peak_gib:.2f}",
                    flush=True,
                )
        acc, mean_iou, ious, precision, recall = metrics(model, val_loader, device, use_amp)
        epoch_seconds = time.perf_counter() - epoch_started
        # Class ids: 1=solid line, 3=curve, 4=point. These are more useful
        # than whole-image accuracy after the user has cropped a region.
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(loader)):.4f} "
            f"val_pixel_acc={acc:.4f} val_mean_iou={mean_iou:.4f} "
            f"line_p={precision[1]:.3f} line_r={recall[1]:.3f} "
            f"point_p={precision[4]:.3f} point_r={recall[4]:.3f} "
            f"curve_p={precision[3]:.3f} curve_r={recall[3]:.3f} "
            f"epoch_s={epoch_seconds:.1f}"
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "classes": CLASSES,
            "size": SIZE,
        }
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(args.out)
        epoch_out = args.out.with_name(f"{args.out.stem}.epoch{epoch + 1}{args.out.suffix}")
        torch.save(payload, epoch_out)
        print(f"saved {args.out} and {epoch_out}", flush=True)
    print(f"training_complete total_s={time.perf_counter() - run_started:.1f}", flush=True)


if __name__ == "__main__":
    main()
