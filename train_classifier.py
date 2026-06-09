"""
train_classifier.py
-------------------
Train MobileNetV2 on your own wet/dry waste photos.

DATASET FOLDER STRUCTURE REQUIRED:
    dataset/
    ├── train/
    │   ├── dry/      ← photos of dry waste (paper, plastic, metal, glass, etc.)
    │   └── wet/      ← photos of wet waste (food scraps, peels, leftovers, etc.)
    └── val/
        ├── dry/
        └── wet/

MINIMUM IMAGES: 10–15 per class is enough for a demo prototype.
MORE IS BETTER: 50+ per class gives much better accuracy.

RUN:
    python train_classifier.py
    python train_classifier.py --epochs 50 --data_dir my_dataset/
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from waste_classifier import build_model, CLASSES

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train wet/dry waste classifier")
parser.add_argument("--data_dir",   default="dataset",           help="Root dataset folder")
parser.add_argument("--output",     default="models/waste_classifier.pt", help="Where to save model")
parser.add_argument("--epochs",     type=int, default=30,        help="Training epochs")
parser.add_argument("--batch_size", type=int, default=8,         help="Batch size (reduce if RAM issue)")
parser.add_argument("--lr",         type=float, default=0.001,   help="Learning rate")
parser.add_argument("--freeze",     action="store_true",          help="Freeze backbone, train head only")
args = parser.parse_args()

# ─── Transforms ───────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Dataset loading ──────────────────────────────────────────────────────────
print(f"\n[Training] Loading dataset from '{args.data_dir}' ...")

train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
val_dataset   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tf)

print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print(f"  Classes found: {train_dataset.classes}")

# Validate class order matches CLASSES
assert train_dataset.classes == CLASSES, (
    f"Expected classes {CLASSES} but found {train_dataset.classes}. "
    "Rename your folders to 'dry' and 'wet' exactly."
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

# ─── Model setup ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = build_model(pretrained=True).to(device)

# Optionally freeze backbone (faster training with small datasets)
if args.freeze:
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    print("  Backbone frozen — training head only")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# ─── Training loop ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
best_val_acc = 0.0

print(f"\n[Training] Starting {args.epochs} epochs ...\n")

for epoch in range(1, args.epochs + 1):
    # --- Train ---
    model.train()
    train_loss, train_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss   += loss.item() * imgs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # --- Validate ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss    += criterion(outputs, labels).item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset) * 100
    val_acc   = val_correct   / len(val_dataset)   * 100

    scheduler.step()

    print(f"Epoch {epoch:3d}/{args.epochs}  "
          f"Train loss: {train_loss/len(train_dataset):.4f}  acc: {train_acc:.1f}%  |  "
          f"Val loss: {val_loss/len(val_dataset):.4f}  acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), args.output)
        print(f"  [OK] Saved best model (val acc: {val_acc:.1f}%)")

print(f"\n[Done] Best validation accuracy: {best_val_acc:.1f}%")
print(f"[Done] Model saved to: {args.output}")