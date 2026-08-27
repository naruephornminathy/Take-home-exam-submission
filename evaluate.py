import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import get_train_val_loaders
from model import UNet

# Metrics

def compute_iou(pred, target, threshold=0.5, smooth=1.0):
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_metrics(pred, target, threshold=0.5, smooth=1e-7):
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)
    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    iou = TP / (TP + FP + FN + smooth)
    f1 = 2 * TP / (2 * TP + FP + FN + smooth)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "iou": iou.item(),
        "f1": f1.item(),
    }


def evaluate_model(model, loader, device):
    model.eval()
    all_metrics = {"accuracy": [], "precision": [], "recall": [], "iou": [], "f1": []}

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_metrics = compute_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)

    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    return avg_metrics

# Qualitative visualization

def get_top_k_iou_indices(model, dataset, device, k=5, min_cloud_ratio=0.05, max_cloud_ratio=0.95):
    """Rank patches by IoU, restricted to patches that actually contain a
    mix of cloud/non-cloud pixels (excludes trivial all-clear or all-cloud
    patches, which otherwise dominate a naive top-k by IoU)."""
    model.eval()
    iou_scores = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, mask = dataset[idx]

            cloud_ratio = mask.sum().item() / mask.numel()
            if cloud_ratio < min_cloud_ratio or cloud_ratio > max_cloud_ratio:
                continue

            image_batch = image.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)
            pred = model(image_batch)
            iou = compute_iou(pred, mask_batch).item()
            iou_scores.append((idx, iou))

    iou_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, score in iou_scores[:k]]
    top_k_scores = [score for idx, score in iou_scores[:k]]

    return top_k_indices, top_k_scores


def save_qualitative_predictions(model, dataset, indices, scores, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for i, idx in enumerate(indices):
        try:
            image, mask = dataset[idx]
            with torch.no_grad():
                pred = model(image.unsqueeze(0).to(device))
                pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float32)

            rgb = np.transpose(image[:3].numpy(), (1, 2, 0))
            rgb = rgb / rgb.max() if rgb.max() > 0 else rgb
            gt = mask.squeeze().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title(f"Prediction (IoU={scores[i]:.3f})")
            axes[2].axis("off")

            plt.tight_layout(pad=2.0)
            plt.savefig(
                os.path.join(save_dir, f"prediction_{i + 1}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Saved patch {i + 1} (idx={idx}, IoU={scores[i]:.3f})")
        except Exception as e:
            print(f"Error at index {idx}: {e}")

    print(f"Saved {len(indices)} qualitative predictions to {save_dir}")


#Entry point

def main():
    parser = argparse.ArgumentParser(description="Evaluate the cloud segmentation U-Net.")
    parser.add_argument("--data_root", type=str, default="./38-cloud")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_qualitative", type=int, default=5)
    parser.add_argument("--metrics_path", type=str, default="outputs/metrics.json")
    parser.add_argument("--predictions_dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, val_loader = get_train_val_loaders(
        args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    print("Evaluation patches (validation split):", len(val_loader.dataset))

    final_metrics = evaluate_model(model, val_loader, device)
    print("Metrics:", final_metrics)

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics to {args.metrics_path}")

    top_indices, top_scores = get_top_k_iou_indices(
        model, val_loader.dataset, device, k=args.num_qualitative
    )
    save_qualitative_predictions(
        model, val_loader.dataset, top_indices, top_scores, device, args.predictions_dir
    )


if __name__ == "__main__":
    main()
