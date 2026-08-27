import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_train_val_loaders
from model import UNet


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (
            1 - self.bce_weight
        ) * self.dice(pred, target)


def compute_iou(pred, target, threshold=0.5, smooth=1.0):
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_f1(pred, target, threshold=0.5, smooth=1.0):
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)
            running_iou += compute_iou(outputs, masks).item() * images.size(0)
            running_f1 += compute_f1(outputs, masks).item() * images.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_iou / n, running_f1 / n


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    num_epochs=50,
    patience=5,
    checkpoint_path="outputs/best_model.pth",
):
    best_val_loss = float("inf")
    best_val_iou = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_iou, val_f1 = validate_one_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_iou={val_iou:.4f} | val_f1={val_f1:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return history


def plot_training_curves(history, save_path="outputs/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(history["val_iou"], label="Val IoU", color="green")
    axes[1].plot(history["val_f1"], label="Val F1", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation IoU and F1")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the cloud segmentation U-Net.")
    parser.add_argument("--data_root", type=str, default="./38-cloud")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default="outputs/best_model.pth")
    parser.add_argument("--curves_path", type=str, default="outputs/training_curves.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet(in_channels=4, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_loader, val_loader = get_train_val_loaders(
        args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    print("Train patches:", len(train_loader.dataset))
    print("Val patches:", len(val_loader.dataset))

    loss_fn = BCEDiceLoss()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
    )

    plot_training_curves(history, save_path=args.curves_path)


if __name__ == "__main__":
    main()
