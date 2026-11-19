# Last-layer finetuning of a pretrained model on a new dataset.

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from utils.transform import transform

def parse_args():
    args = argparse.ArgumentParser(description="Last-layer finetuning of a pretrained model on a new dataset.")
    args.add_argument("--epochs", type=int, default=10, help="Number of epochs for finetuning.")
    args.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    args.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    args.add_argument("--l2_reg", type=float, default=0.0, help="L2 regularization strength.")
    args.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization strength.")
    args.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    return args.parse_args()


def train_model(model, train_loader, val_loader, loss_function, optimizer, device, epochs, lr_scheduler):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")