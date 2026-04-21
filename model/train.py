import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

DATA_DIR = r"data\PlantVillage"
MODEL_DIR = os.environ.get("MODEL_DIR", "./saved")
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 224
NUM_WORKERS = 4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_datasets(data_dir: str):
    train_tf, val_tf = get_transforms()
    full = datasets.ImageFolder(data_dir, transform=train_tf)
    n = len(full)
    n_test = int(n * TEST_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset = copy.copy(full)
    val_ds.dataset.transform = val_tf
    test_ds.dataset = copy.copy(full)
    test_ds.dataset.transform = val_tf
    return train_ds, val_ds, test_ds, full.classes


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    train_ds, val_ds, test_ds, classes = load_datasets(DATA_DIR)
    print(f"Classes: {len(classes)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(classes, f)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl, va = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] | {elapsed:.1f}s | "
              f"Train Loss: {tl:.4f} Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f} Acc: {va:.4f}")

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_weights.pth"))
            print(f"  -> Saved best model (val_acc={va:.4f})")

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch, eta_min=1e-7)
            print("  -> Unfrozen full model for fine-tuning")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_weights.pth")))

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(classes),
        "class_names": classes,
        "test_acc": test_acc,
        "history": history,
    }, os.path.join(MODEL_DIR, "plant_disease_model.pt"))

    with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nModel saved to {MODEL_DIR}/plant_disease_model.pt")


if __name__ == "__main__":
    main()