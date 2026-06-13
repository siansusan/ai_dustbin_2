# quick_eval.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from waste_classifier import build_model, CLASSES

model = build_model(pretrained=False)
model.load_state_dict(torch.load("models/waste_classifier.pt", map_location="cpu"))
model.eval()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Determine validation directory path dynamically
val_dir = "dataset/val"
for possible_path in ["dataset/val", "dataset/valid", "val", "valid"]:
    if os.path.exists(possible_path):
        val_dir = possible_path
        break

print(f"Loading validation data from '{val_dir}' ...")
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
loader = DataLoader(val_data, batch_size=32)

correct, total = 0, 0
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nVal accuracy: {correct/total*100:.2f}%  ({correct}/{total} correct)")
print(f"Classes: {val_data.classes}")