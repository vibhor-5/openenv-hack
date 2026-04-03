"""
Dataset loading and subsetting for the ML Training Optimizer environment.

Provides deterministic subsets of standard datasets:
- MNIST: 5k samples (4k train / 1k val)
- FashionMNIST: 8k samples (6.5k train / 1.5k val)
- CIFAR-10: 10k samples (8k train / 2k val)

Small subsets intentionally make overfitting a real challenge.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms


# Cache datasets inside the container
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/ml_trainer_data")


def _get_mnist_transforms(augment: bool = False, aug_strength: float = 0.5):
    """Get transforms for MNIST (28×28 grayscale)."""
    base = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    if augment:
        # Scale rotation and translation by augmentation strength
        max_rot = int(15 * aug_strength)
        max_translate = 0.1 * aug_strength
        aug = [
            transforms.RandomRotation(max_rot),
            transforms.RandomAffine(0, translate=(max_translate, max_translate)),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def _get_fashion_transforms(augment: bool = False, aug_strength: float = 0.5):
    """Get transforms for FashionMNIST (28×28 grayscale)."""
    base = [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    if augment:
        max_rot = int(10 * aug_strength)
        aug = [
            transforms.RandomHorizontalFlip(p=0.5 * aug_strength),
            transforms.RandomRotation(max_rot),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def _get_cifar_transforms(augment: bool = False, aug_strength: float = 0.5):
    """Get transforms for CIFAR-10 (32×32 RGB)."""
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    if augment:
        crop_pad = max(1, int(4 * aug_strength))
        aug = [
            transforms.RandomCrop(32, padding=crop_pad),
            transforms.RandomHorizontalFlip(p=0.5 * aug_strength),
        ]
        if aug_strength > 0.5:
            aug.append(
                transforms.ColorJitter(
                    brightness=0.2 * aug_strength,
                    contrast=0.2 * aug_strength,
                )
            )
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


def load_dataset(
    dataset_name: str,
    seed: int = 42,
    augment: bool = False,
    aug_strength: float = 0.5,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, int]:
    """
    Load a dataset and return deterministic train/val subsets.

    Args:
        dataset_name: One of 'mnist', 'fashion_mnist', 'cifar10'
        seed: Random seed for reproducible subsetting
        augment: Whether to apply data augmentation to training set
        aug_strength: Augmentation intensity (0.0 to 1.0)

    Returns:
        (train_dataset, val_dataset, num_train, num_val)
    """
    generator = torch.Generator().manual_seed(seed)

    if dataset_name == "mnist":
        transform_train = _get_mnist_transforms(augment, aug_strength)
        transform_val = _get_mnist_transforms(augment=False)
        full_train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform_train)
        full_val = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform_val)
        total_subset = 5000
        train_size, val_size = 4000, 1000

    elif dataset_name == "fashion_mnist":
        transform_train = _get_fashion_transforms(augment, aug_strength)
        transform_val = _get_fashion_transforms(augment=False)
        full_train = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform_train)
        full_val = torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform_val)
        total_subset = 8000
        train_size, val_size = 6500, 1500

    elif dataset_name == "cifar10":
        transform_train = _get_cifar_transforms(augment, aug_strength)
        transform_val = _get_cifar_transforms(augment=False)
        full_train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
        full_val = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_val)
        total_subset = 10000
        train_size, val_size = 8000, 2000

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: mnist, fashion_mnist, cifar10")

    # Create deterministic subset indices
    all_indices = torch.randperm(len(full_train), generator=generator)[:total_subset].tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    return train_dataset, val_dataset, train_size, val_size


def create_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    seed: int = 42,
    augment: bool = False,
    aug_strength: float = 0.5,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create DataLoaders for a dataset.

    Returns:
        (train_loader, val_loader, num_train, num_val)
    """
    train_ds, val_ds, n_train, n_val = load_dataset(
        dataset_name, seed=seed, augment=augment, aug_strength=aug_strength
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep it simple for 2 vCPU
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # Larger batch for eval (no grads)
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, n_train, n_val


def download_all_datasets():
    """Pre-download all datasets. Called during Docker build."""
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)
    print("Downloading FashionMNIST...")
    torchvision.datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    print("All datasets downloaded.")


if __name__ == "__main__":
    download_all_datasets()
