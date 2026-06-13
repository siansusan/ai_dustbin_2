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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from waste_classifier import build_model, CLASSES

# ─── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train wet/dry waste classifier")
parser.add_argument("--data_dir",   default=None,                help="Root dataset folder (defaults to '.' if train/val exist)")
parser.add_argument("--output",     default="models/waste_classifier.pt", help="Where to save model")
parser.add_argument("--epochs",     type=int, default=30,        help="Training epochs")
parser.add_argument("--batch_size", type=int, default=8,         help="Batch size (reduce if RAM issue)")
parser.add_argument("--lr",         type=float, default=0.001,   help="Learning rate")
parser.add_argument("--freeze",     action="store_true",          help="Freeze backbone, train head only")
parser.add_argument("--only_new",   action="store_true",          help="Train only on the new images (prefixed with 'dataset2_')")
args = parser.parse_args()

# Dynamic dataset directory detection
if args.data_dir is None:
    if os.path.exists("train") and (os.path.exists("val") or os.path.exists("valid")):
        args.data_dir = "."
    else:
        args.data_dir = "dataset"

# Detect validation directory name
val_dir_name = "val"
if not os.path.exists(os.path.join(args.data_dir, "val")) and os.path.exists(os.path.join(args.data_dir, "valid")):
    val_dir_name = "valid"

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

train_path = os.path.join(args.data_dir, "train")
val_path   = os.path.join(args.data_dir, val_dir_name)

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train folder not found at: {train_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation folder not found at: {val_path}")

train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
val_dataset   = datasets.ImageFolder(val_path,   transform=val_tf)

if args.only_new:
    train_dataset.samples = [s for s in train_dataset.samples if os.path.basename(s[0]).startswith("dataset2_")]
    train_dataset.imgs = train_dataset.samples
    val_dataset.samples = [s for s in val_dataset.samples if os.path.basename(s[0]).startswith("dataset2_")]
    val_dataset.imgs = val_dataset.samples
    print("  [Filtered] Training only on newly merged images (prefixed with 'dataset2_')")

print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print(f"  Classes found: {train_dataset.classes}")

# Validate class order: first class should represent dry waste, second should represent wet waste
assert len(train_dataset.classes) == 2, f"Expected exactly 2 classes, found {len(train_dataset.classes)}: {train_dataset.classes}"
assert "dry" in train_dataset.classes[0].lower(), f"Expected first class folder to be 'dry' or 'dry waste', got '{train_dataset.classes[0]}'"
assert "wet" in train_dataset.classes[1].lower(), f"Expected second class folder to be 'wet' or 'wet waste', got '{train_dataset.classes[1]}'"

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

# Calculate class weights to handle imbalance
targets = train_dataset.targets
class_counts = [targets.count(0), targets.count(1)]
print(f"  Class counts in train: dry={class_counts[0]}, wet={class_counts[1]}")

if class_counts[0] > 0 and class_counts[1] > 0:
    total_samples = sum(class_counts)
    weights = [total_samples / (2.0 * count) for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Loss function: weighted CrossEntropyLoss(dry={weights[0]:.3f}, wet={weights[1]:.3f})")
else:
    criterion = nn.CrossEntropyLoss()
    print("  Loss function: standard CrossEntropyLoss")

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