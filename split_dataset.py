"""
split_dataset.py
----------------
Splits your train folder into train + val (80/20 split).
Run this from the folder that CONTAINS your 'train' folder.

Usage: python split_dataset.py
"""

import os
import shutil
import random

TRAIN_DIR = "train"
VAL_DIR   = "val"
VAL_RATIO = 0.2      # 20% goes to val
SEED      = 42

random.seed(SEED)

for class_name in ["dry waste", "wet waste"]:
    src = os.path.join(TRAIN_DIR, class_name)
    if not os.path.exists(src):
        print(f"Folder not found: {src}")
        continue

    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    val_count = max(1, int(len(images) * VAL_RATIO))
    val_images = images[:val_count]

    val_dst = os.path.join(VAL_DIR, class_name)
    os.makedirs(val_dst, exist_ok=True)

    for img in val_images:
        shutil.move(os.path.join(src, img), os.path.join(val_dst, img))

    print(f"[{class_name}] {len(images) - val_count} train | {val_count} val")

print("\nDone! Your dataset is ready.")
print("Run: python train_classifier.py --data_dir .")