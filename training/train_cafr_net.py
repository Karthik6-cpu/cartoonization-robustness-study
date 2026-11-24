"""Train CAFR-Net with matched clean/shifted training views.

Expected layout:
  DATA_ROOT/train/<class_name>/*.(jpg|jpeg|png)
  DATA_ROOT/val/<class_name>/*.(jpg|jpeg|png)

This generic starter uses train-only JPEG/blur/brightness/noise shifts.
For the final paper, add the verified cartoonization pipelines to TRAIN
identities only; never create training views from validation/test identities.
"""
from __future__ import annotations
import argparse, io, json, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.cafr_net import CAFRNet, paired_consistency_loss


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def shifted_view(image: Image.Image) -> Image.Image:
    """Sample a mild train-only appearance/compression perturbation."""
    choice = random.choice(("jpeg", "blur", "brightness", "noise", "identity"))
    if choice == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    if choice == "brightness":
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    if choice == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.choice((35, 50, 65, 80)))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if choice == "noise":
        arr = np.asarray(image).astype(np.float32)
        arr += np.random.normal(0, random.uniform(3, 12), arr.shape)
        return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    return image.copy()


class PairedImageFolder(Dataset):
    def __init__(self, root: str, image_size: int = 224):
        self.base = ImageFolder(root)
        self.geometry = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.tensor = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        path, label = self.base.samples[index]
        image = self.base.loader(path).convert("RGB")
        clean = self.geometry(image)
        shifted = shifted_view(clean)
        return self.tensor(clean), self.tensor(shifted), label


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Root containing train/ and val/")
    p.add_argument("--out", default="outputs/cafr_net")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lambda-consistency", type=float, default=0.25)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = PairedImageFolder(str(root / "train"), args.image_size)
    val_ds = ImageFolder(str(root / "val"), transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ]))
    if train_ds.base.classes != val_ds.classes:
        raise ValueError("Train/validation class folders do not match.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = CAFRNet(num_classes=len(train_ds.base.classes), pretrained=True).to(device)
    # Warm-up: freeze the pretrained encoder while new modules adapt.
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("encoder.")], "lr": 2e-4},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()
    best_acc, patience, stale = -1.0, 5, 0
    history = []

    for epoch in range(args.epochs):
        if epoch == 2:
            for param in model.encoder.parameters():
                param.requires_grad = True
        model.train()
        running = total = 0
        for clean, shifted, labels in train_loader:
            clean, shifted, labels = clean.to(device), shifted.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits_clean = model(clean)
                logits_shift = model(shifted)
                loss = (ce(logits_clean, labels) + ce(logits_shift, labels)
                        + args.lambda_consistency *
                        paired_consistency_loss(logits_clean, logits_shift))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += loss.item() * labels.size(0); total += labels.size(0)

        metrics = validate(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": running / max(total, 1), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["accuracy"] > best_acc:
            best_acc, stale = metrics["accuracy"], 0
            torch.save({
                "model": model.state_dict(), "classes": train_ds.base.classes,
                "seed": args.seed, "epoch": epoch + 1,
                "val_accuracy": best_acc, "args": vars(args),
            }, out / "best_cafr_net.pt")
        else:
            stale += 1
        if stale >= patience:
            break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Checkpoint: {out / 'best_cafr_net.pt'}")


if __name__ == "__main__":
    main()
