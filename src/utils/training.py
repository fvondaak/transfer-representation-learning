import csv

import matplotlib.pyplot as plt
import torch

HISTORY_FIELDS = ["epoch", "val_acc", "val_loss", "train_acc", "train_loss"]

def get_device():
    # AMD GPUs (ROCm builds of PyTorch) are also exposed as "cuda"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct_predictions += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, correct_predictions / total_samples

def save_history(history, path):
    """Write the history (list of dicts) to <path>.csv and plot it to <path>.png."""
    with open(f"{path}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("loss", "acc")):
        for split in ("train", "val"):
            ax.plot([row["epoch"] for row in history], [row[f"{split}_{metric}"] for row in history], label=split)
        ax.set(xlabel="epoch", ylabel=metric)
        ax.legend()
    fig.savefig(f"{path}.png")
    plt.close(fig)

def evaluation(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy