# Full finetuning (all layers are optimized) of a pretrained ResNet-50.
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.data import ChestXrayDataset, get_test_dataloader, get_train_dataloader, get_val_dataloader
from utils.training import evaluation, get_device, save_history, train_epoch
from utils.transform import stretch_and_normalize

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Full finetuning of a pretrained ResNet-50.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--l2_reg", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--model_name", default="resnet50_imagenet.pth", help="Pretrained model file in models/.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for data loading.")
    return parser.parse_args()


def load_model(model_name):
    model = resnet50()
    model.load_state_dict(torch.load(MODELS_DIR / model_name, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, len(ChestXrayDataset.classes))
    return model


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    loader_args = dict(batch_size=args.batch_size, transform=stretch_and_normalize, num_workers=args.num_workers)
    train_loader = get_train_dataloader(**loader_args)
    val_loader = get_val_dataloader(**loader_args)
    test_loader = get_test_dataloader(**loader_args)

    model = load_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_reg)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluation(model, val_loader, criterion, device)
        history.append(dict(epoch=epoch, val_acc=val_acc, val_loss=val_loss, train_acc=train_acc, train_loss=train_loss))
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "resnet_50_full.pth")
    RESULTS_DIR.mkdir(exist_ok=True)
    save_history(history, RESULTS_DIR / f"resnet_50_full_lr{args.learning_rate}_l2{args.l2_reg}_bs{args.batch_size}")

    test_loss, test_acc = evaluation(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
