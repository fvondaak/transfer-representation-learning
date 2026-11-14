import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from torchvision.datasets import CIFAR10

from datasets import load_dataset

from medmnist import PathMNIST

dataset = load_dataset("ILSVRC/imagenet-1k")

# ImageNet Dataset used to train the ResNet50 model
class ImageNetDataset(Dataset):
    def __init__(self, split, transform = None):
        self.dataset = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

class PathMNISTDataset(Dataset):
    def __init__(self, split, transform=None):
        self.dataset = PathMNIST(split=split, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_train_dataloader(batch_size=32, num_workers=4, transform=None, shuffle=True, dataset_name="imagenet"):

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(split="train", transform=transform)
    elif dataset_name == "pathmnist":
        train_dataset = PathMNISTDataset(split="train", transform=transform)
    else:
        raise ValueError("Unsupported dataset name")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader

def get_test_dataloader(batch_size=32, num_workers=4, transform=None, shuffle=False, dataset_name="imagenet"):
    "Uses test set from imagenet-1k dataset, does not have labels"

    if dataset_name == "imagenet":
        test_dataset = ImageNetDataset(split="test", transform=transform)
    elif dataset_name == "pathmnist":
        test_dataset = PathMNISTDataset(split="test", transform=transform)
    else:
        raise ValueError("Unsupported dataset name")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return test_loader

def get_val_dataloader(batch_size=32, num_workers=4, transform=None, shuffle=False, dataset_name="imagenet"):

    if dataset_name == "imagenet":
        val_dataset = ImageNetDataset(split="validation", transform=transform)
    elif dataset_name == "pathmnist":
        val_dataset = PathMNISTDataset(split="validation", transform=transform)
    else:
        raise ValueError("Unsupported dataset name")

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return val_loader