"""Data loading utilities for CIFAR-10."""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def get_cifar10_dataloader(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
    train: bool = True,
) -> DataLoader:
    """Get CIFAR-10 dataloader.
    
    Args:
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        data_root: Root directory for storing/loading the dataset
        train: Whether to load training set (True) or test set (False)
        
    Returns:
        DataLoader for CIFAR-10
    """
    # Simple transforms: just normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,  # Drop last incomplete batch only for training
    )
    
    return dataloader


def get_cifar10_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """Get both train and test dataloaders for CIFAR-10.
    
    Args:
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes
        data_root: Root directory for the dataset
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = get_cifar10_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
        train=True,
    )
    
    test_loader = get_cifar10_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
        train=False,
    )
    
    return train_loader, test_loader

