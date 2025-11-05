import os
import json
import torch
from utils.dataset import get_dataloaders
from utils.trainer import train_model
from models.plain_cnn import PlainCNN
from models.hybrid_cnn import HybridCNNQNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(data_dir="Assets/Leather Defect Classification", epochs=8, batch_size=16, lr=1e-3):
    train_dl, val_dl, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)

    results = {}

    # Train Plain CNN
    plain_ckpt = "plain_cnn.pth"
    if not os.path.exists(plain_ckpt):
        print("Training Plain CNN...")
        plain_model = PlainCNN(num_classes)
        plain_acc = train_model(plain_model, train_dl, val_dl, epochs, lr, DEVICE, plain_ckpt)
        results["plain_val_acc"] = plain_acc
    else:
        print("Found existing PlainCNN checkpoint, skipping training.")
        results["plain_val_acc"] = torch.load(plain_ckpt)["val_acc"]

    # Train Hybrid CNN
    hybrid_ckpt = "hybrid_cnn.pth"
    if not os.path.exists(hybrid_ckpt):
        print("Training Hybrid CNN...")
        hybrid_model = HybridCNNQNN(num_classes)
        hybrid_acc = train_model(hybrid_model, train_dl, val_dl, epochs, lr, DEVICE, hybrid_ckpt)
        results["hybrid_val_acc"] = hybrid_acc
    else:
        print("Found existing HybridCNN checkpoint, skipping training.")
        results["hybrid_val_acc"] = torch.load(hybrid_ckpt)["val_acc"]

    hybrid_model = HybridCNNQNN(num_classes)
    hybrid_model.load_state_dict(torch.load(hybrid_ckpt)["model_state_dict"])
    plot_confusion_matrix(hybrid_model, val_dl, classes, DEVICE, "hybrid_confusion_matrix.png")

    plain_model = PlainCNN(num_classes)
    plain_model.load_state_dict(torch.load(plain_ckpt)["model_state_dict"])
    plot_confusion_matrix(plain_model, val_dl, classes, DEVICE, "plain_confusion_matrix.png")
    
    # Save results
    results["classes"] = classes
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Training complete:", results)

if __name__ == "__main__":
    main()
