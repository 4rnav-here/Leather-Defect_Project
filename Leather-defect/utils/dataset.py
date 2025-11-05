import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

class TransformedSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders(data_dir, batch_size=16, num_workers=4, seed=42):
    random.seed(seed)
    base = datasets.ImageFolder(root=data_dir, transform=None)
    total = len(base)
    val_len = max(1, int(0.2 * total))
    train_len = total - val_len

    indices = list(range(total))
    random.shuffle(indices)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]

    train_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = TransformedSubset(base, train_idx, train_tf)
    val_ds = TransformedSubset(base, val_idx, val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, base.classes
