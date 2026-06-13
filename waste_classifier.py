"""
waste_classifier.py
-------------------
MobileNetV2-based binary classifier for wet vs dry waste.
Runs on Raspberry Pi (CPU) in ~150-200ms per inference.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import time


# ─── Label map ────────────────────────────────────────────────────────────────
CLASSES = ["dry waste", "wet waste"]          # index 0 = dry, index 1 = wet
NUM_CLASSES = 2


# ─── Image preprocessing (must match training transforms) ─────────────────────
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model definition ─────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    MobileNetV2 with custom final layer for binary classification.
    Pretrained ImageNet weights used as base — fine-tuned on your dataset.
    """
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    )
    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ─── WasteClassifier class ────────────────────────────────────────────────────
class WasteClassifier:
    """
    Loads a trained MobileNetV2 model and classifies a captured image
    as 'wet' or 'dry'. Returns class label + confidence score.
    """

    def __init__(self, model_path: str = "models/waste_classifier.pt", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = build_model(pretrained=False)  # Don't re-download weights at inference
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"[WasteClassifier] Model loaded from '{model_path}' on {self.device}")

    def classify_image(self, image) -> dict:
        """
        Args:
            image: PIL Image OR numpy array (BGR from OpenCV)
        Returns:
            {"label": "wet"/"dry", "confidence": 0.0–1.0, "inference_ms": float}
        """
        # Convert numpy/BGR to PIL/RGB
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        inference_ms = (time.time() - t0) * 1000

        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())

        return {
            "label": CLASSES[class_idx],
            "confidence": confidence,
            "inference_ms": round(inference_ms, 1),
            "dry_prob": round(float(probs[0].item()), 3),
            "wet_prob": round(float(probs[1].item()), 3),
        }

    def classify_from_camera(self, camera_index: int = 0) -> dict:
        """Capture a single frame from camera and classify it."""
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return self.classify_image(frame)


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print("Model output shape:", out.shape)  # Should be [1, 2]
    print("Classes:", CLASSES)
    print("Model OK ✓")