import os
import json
import sys
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "../model/saved/plant_disease_model.pt")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "../model/class_names.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

_model = None
_class_names = None


def _build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model():
    global _model, _class_names
    if _model is not None:
        return _model, _class_names

    if not os.path.exists(MODEL_PATH):
        with open(CLASS_NAMES_PATH) as f:
            _class_names = json.load(f)
        print(f"WARNING: Model file not found at {MODEL_PATH}. Running in demo mode.")
        _model = None
        return _model, _class_names

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    _class_names = checkpoint["class_names"]
    _model = _build_model(len(_class_names))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(DEVICE).eval()
    print(f"Model loaded: {len(_class_names)} classes on {DEVICE}")
    return _model, _class_names


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def parse_class_name(raw: str) -> tuple[str, str]:
    parts = raw.split("___")
    plant = parts[0].replace("_", " ").replace(",", "").strip().title()
    if len(parts) > 1:
        disease_raw = parts[1].replace("_", " ").strip()
        disease = disease_raw if disease_raw.lower() != "healthy" else "Healthy"
    else:
        disease = "Unknown"
    return plant, disease


@torch.no_grad()
def predict(image: Image.Image, use_gradcam: bool = False) -> dict:
    model, class_names = load_model()

    if model is None:
        import random
        idx = random.randint(0, len(class_names) - 1)
        raw = class_names[idx]
        plant, disease = parse_class_name(raw)
        probs = [0.01] * len(class_names)
        probs[idx] = 0.82
        top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
        return {
            "class_index": idx,
            "raw_class": raw,
            "plant": plant,
            "disease": disease,
            "is_healthy": disease == "Healthy",
            "confidence": 0.82,
            "top5": [{"class": class_names[i], "confidence": round(p, 4)} for i, p in top5],
            "gradcam_image": None,
            "demo_mode": True,
        }

    tensor = preprocess_image(image)
    output = model(tensor)
    probs = F.softmax(output, dim=1)[0]
    top5_vals, top5_idxs = torch.topk(probs, 5)

    idx = top5_idxs[0].item()
    confidence = top5_vals[0].item()
    raw = class_names[idx]
    plant, disease = parse_class_name(raw)

    top5 = [
        {"class": class_names[i.item()], "confidence": round(v.item(), 4)}
        for i, v in zip(top5_idxs, top5_vals)
    ]

    gradcam_b64 = None
    if use_gradcam:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
            from gradcam import get_gradcam_b64
            gradcam_b64 = get_gradcam_b64(model, preprocess_image(image), image, idx)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    return {
        "class_index": idx,
        "raw_class": raw,
        "plant": plant,
        "disease": disease,
        "is_healthy": disease == "Healthy",
        "confidence": round(confidence, 4),
        "top5": top5,
        "gradcam_image": gradcam_b64,
        "demo_mode": False,
    }