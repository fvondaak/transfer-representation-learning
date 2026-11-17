from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

ROOT = Path(__file__).resolve().parent.parent.parent
DATAPATH = ROOT / "data"

class ChestXrayDataset(Dataset):
    
    classes = ("NORMAL", "PNEUMONIA")

    def __init__(self, root_dir=DATAPATH, transform=None, split="train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {split!r}; expected 'train', 'val', or 'test'.")

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        dataset_dir = self.root_dir
        if not (dataset_dir / split).is_dir():
            dataset_dir = dataset_dir / "chest_xray"
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

        self.samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Dataset class directory not found: {class_dir}")
            self.samples.extend(
                (path, label)
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in (".jpeg", ".jpg", ".png")
                and not path.name.startswith(".")
            )
        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")
        self.image_paths = [path for path, _ in self.samples]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = torchvision.io.read_image(
            str(img_path), mode=torchvision.io.ImageReadMode.RGB
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_train_dataloader(batch_size=32, transform=None, shuffle=True):
    dataset = ChestXrayDataset(split="train", transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_test_dataloader(batch_size=32, transform=None, shuffle=False):
    dataset = ChestXrayDataset(split="test", transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_val_dataloader(batch_size=32, transform=None, shuffle=False):
    dataset = ChestXrayDataset(split="val", transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)