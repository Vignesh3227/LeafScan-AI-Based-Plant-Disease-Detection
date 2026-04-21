import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.environ.get("MODEL_PATH", "./saved/plant_disease_model.pt")
DATA_DIR = os.environ.get("DATA_DIR", "../data/PlantVillage")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./evaluation")
BATCH_SIZE = 64
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes):
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


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = build_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, checkpoint["class_names"]


def get_test_loader(data_dir, class_names):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = datasets.ImageFolder(data_dir, transform=tf)
    n = len(full)
    n_test = int(n * 0.10)
    n_rest = n - n_test
    _, test_ds = random_split(full, [n_rest, n_test], generator=torch.Generator().manual_seed(42))
    return DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def predict_all(model, loader):
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, class_names, output_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Plant Disease Detection", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrix.png")


def plot_training_history(history_path, output_dir):
    if not os.path.exists(history_path):
        return
    with open(history_path) as f:
        history = json.load(f)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    ax1.set_title("Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Validation", linewidth=2)
    ax2.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved training_history.png")


def plot_per_class_accuracy(labels, preds, class_names, output_dir):
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    sorted_idx = np.argsort(per_class_acc)
    fig, ax = plt.subplots(figsize=(10, 14))
    bars = ax.barh(
        [class_names[i].replace("___", "\n").replace("_", " ") for i in sorted_idx],
        per_class_acc[sorted_idx],
        color=plt.cm.RdYlGn(per_class_acc[sorted_idx])
    )
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(x=per_class_acc.mean(), color="navy", linestyle="--", label=f"Mean: {per_class_acc.mean():.3f}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved per_class_accuracy.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    model, class_names = load_model(MODEL_PATH)
    print(f"Loaded model with {len(class_names)} classes")

    test_loader = get_test_loader(DATA_DIR, class_names)
    print(f"Test samples: {len(test_loader.dataset)}")

    labels, preds = predict_all(model, test_loader)
    acc = (labels == preds).mean()
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(labels, preds, class_names, OUTPUT_DIR)
    plot_training_history(os.path.dirname(MODEL_PATH) + "/history.json", OUTPUT_DIR)
    plot_per_class_accuracy(labels, preds, class_names, OUTPUT_DIR)

    print(f"\nAll evaluation artifacts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()