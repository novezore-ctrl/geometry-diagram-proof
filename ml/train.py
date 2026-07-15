"""Train the compact segmentation model on image + .mask.png pairs."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from model import GeometryUNet, CLASSES


class GeometryDataset(Dataset):
    def __init__(self, root: Path):
        self.images = sorted(p for p in root.glob("*.png") if not p.name.endswith(".mask.png"))
    def __len__(self): return len(self.images)
    def __getitem__(self, index):
        image = TF.to_tensor(Image.open(self.images[index]).convert("RGB"))
        mask = torch.from_numpy(__import__("numpy").array(Image.open(self.images[index].with_name(self.images[index].stem + ".mask.png")), dtype="int64"))
        return image, mask


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = GeometryDataset(args.data); loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    model = GeometryUNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3); loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device); optimizer.zero_grad(); loss = loss_fn(model(images), masks); loss.backward(); optimizer.step(); total += loss.detach().item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(1, len(loader)):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "classes": CLASSES}, args.out); print(f"saved {args.out}")


if __name__ == "__main__": main()
