import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.dataset import get_dataloaders
from utils.trainer import train_model
from utils.metrics import plot_confusion_matrix
from models.plain_cnn import PlainCNN
from models.hybrid_cnn import HybridCNNQNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, loader, classes, device):
    """Compute accuracy, confusion matrix, and classification report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=classes, output_dict=True
    )
    acc = np.trace(cm) / np.sum(cm)
    return acc, cm.tolist(), report


def safe_load_checkpoint(model, checkpoint_path):
    """Safely load checkpoint if compatible."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded existing checkpoint: {checkpoint_path}")
        return ckpt
    except Exception as e:
        print(f"⚠️ Checkpoint mismatch ({e}), retraining from scratch.")
        return None


def main(data_dir="Assets/Leather Defect Classification", epochs=8, batch_size=16, lr=1e-3):
    # Prepare data
    train_dl, val_dl, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)
    print(f"Detected classes: {classes}")

    results = {"classes": classes}

    # ----- Train Plain CNN -----
    plain_ckpt = "plain_cnn.pth"
    plain_model = PlainCNN(num_classes)

    ckpt = safe_load_checkpoint(plain_model, plain_ckpt)
    if ckpt is None:
        print("🚀 Training Plain CNN...")
        plain_acc = train_model(plain_model, train_dl, val_dl, epochs, lr, DEVICE, plain_ckpt)
    else:
        plain_acc = ckpt.get("val_acc", 0.0)
    results["plain_val_acc"] = plain_acc

    # ----- Train Hybrid CNN -----
    hybrid_ckpt = "hybrid_cnn.pth"
    hybrid_model = HybridCNNQNN(num_classes)

    ckpt = safe_load_checkpoint(hybrid_model, hybrid_ckpt)
    if ckpt is None:
        print("🚀 Training Hybrid CNN...")
        hybrid_acc = train_model(hybrid_model, train_dl, val_dl, epochs, lr, DEVICE, hybrid_ckpt)
    else:
        hybrid_acc = ckpt.get("val_acc", 0.0)
    results["hybrid_val_acc"] = hybrid_acc

    # ----- Evaluate and Save Metrics -----
    print("\n📊 Evaluating models...")
    plain_model.load_state_dict(torch.load(plain_ckpt)["model_state_dict"])
    hybrid_model.load_state_dict(torch.load(hybrid_ckpt)["model_state_dict"])

    plain_acc, plain_cm, plain_report = evaluate_model(plain_model, val_dl, classes, DEVICE)
    hybrid_acc, hybrid_cm, hybrid_report = evaluate_model(hybrid_model, val_dl, classes, DEVICE)

    results["plain"] = {
        "val_acc": float(plain_acc),
        "confusion_matrix": plain_cm,
        "report": plain_report
    }
    results["hybrid"] = {
        "val_acc": float(hybrid_acc),
        "confusion_matrix": hybrid_cm,
        "report": hybrid_report
    }

    # ----- Save Confusion Matrices -----
    plot_confusion_matrix(plain_model, val_dl, classes, DEVICE, "plain_confusion_matrix.png")
    plot_confusion_matrix(hybrid_model, val_dl, classes, DEVICE, "hybrid_confusion_matrix.png")

    # ----- Save Metrics -----
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Training and evaluation complete. Results saved to metrics.json")
    print(json.dumps({
        "plain_acc": plain_acc,
        "hybrid_acc": hybrid_acc
    }, indent=2))


if __name__ == "__main__":
    main()
