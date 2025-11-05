import torch
from torchvision import datasets

# Path to your dataset
DATA_DIR = "Assets/Leather Defect Classification"

# Get the correct class list from dataset
dataset = datasets.ImageFolder(root=DATA_DIR)
classes = dataset.classes
print("Detected classes:", classes)

# Update checkpoints
for ckpt_name in ["plain_cnn.pth", "hybrid_cnn.pth"]:
    try:
        ckpt = torch.load(ckpt_name, map_location="cpu")
        ckpt["classes"] = classes
        torch.save(ckpt, ckpt_name)
        print(f"✅ Updated {ckpt_name} with classes: {classes}")
    except Exception as e:
        print(f"⚠️ Could not update {ckpt_name}: {e}")
