import torch
import torch.nn as nn
from utils.metrics import accuracy

def train_model(model, train_loader, val_loader, epochs, lr, device, checkpoint_path):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_acc = accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val
            }, checkpoint_path)
    return best_val
