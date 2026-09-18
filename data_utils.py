"""
data_utils.py

Helpers to load the already-split dataset (train / dev / validation)
into PyTorch DataLoaders, ready for training and evaluation.

Expected folder layout (produced by prepare_data_split.py):

    dataset_split/
        train/<class>/...
        dev/<class>/...
        validation/<class>/*.jpg

Usage:
    from data_utils import get_dataloaders
    train_loader, dev_loader, val_loader = get_dataloaders()

Run directly to see a quick sanity check:
    python data_utils.py
"""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from base_model import ImageNetSubset
from build_augmentation import AugmentationConfig, AugmentationFactory


# Default folder where prepare_data_split.py wrote the split.
DEFAULT_DATA_ROOT = Path("dataset_split")

# ImageNet normalization stats. Must match what evaluate.py uses at
# grading time, otherwise predictions will be wrong.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Standard image size for ImageNet-style models.
IMAGE_SIZE = 224


def build_train_transform() -> transforms.Compose:
    """Transform used for the train split (with light augmentation)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform() -> transforms.Compose:
    """Transform used for dev and validation (no augmentation).

    This mirrors the transform inside evaluate.py exactly, so dev
    accuracy is a fair estimate of the final grading accuracy.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    train_transform: Optional[Callable[[Any], Any]] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build the three DataLoaders.

    Train loader: shuffled, augmented (see `train_transform`).
    Dev and validation loaders: NOT shuffled, NEVER augmented.

    Args:
        data_root: folder containing train/, dev/, validation/.
        batch_size: batch size shared by all three loaders.
        num_workers: number of DataLoader worker processes.
        pin_memory: pinned memory flag. Auto-detected when None.
        train_transform: callable applied to each train image.
            If None, falls back to `build_train_transform()`.
            Dev and validation always use `build_eval_transform()`,
            this argument cannot change that.

    Returns:
        (train_loader, dev_loader, val_loader)
    """
    data_root = Path(data_root)

    # Train can be augmented; dev and validation are LOCKED to the clean
    # eval transform so dev numbers always mirror evaluate.py exactly.
    effective_train_transform = (
        train_transform if train_transform is not None else build_train_transform()
    )

    # Use the dataset class from base_model.py so the label mapping
    # (class folder name -> 0..19 local index) is exactly the same as
    # what evaluate.py and the grader use.
    train_set = ImageNetSubset(
        data_root, split="train", transform=effective_train_transform,
    )
    dev_set = ImageNetSubset(
        data_root, split="dev", transform=build_eval_transform(),
    )
    val_set = ImageNetSubset(
        data_root, split="validation", transform=build_eval_transform(),
    )

    # Use pinned memory automatically when CUDA is available.
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, dev_loader, val_loader


def get_stress_test_loader(
    data_root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 2,
    config: AugmentationConfig | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """
    Build a DataLoader for robustness evaluation.

    The transform comes from AugmentationFactory(config).stress_test(),
    which applies moderate corruptions (color/blur/noise/erasing/etc.) to
    each validation image. Useful as a diagnostic AFTER training; do NOT
    use this during training.

    Args:
        data_root: folder containing the validation/ split.
        batch_size, num_workers, pin_memory: passed through to DataLoader.
        config: optional AugmentationConfig. If None, defaults are used.

    Returns:
        A DataLoader over the validation split, shuffle=False.
    """
    data_root = Path(data_root)
    aug_config = config or AugmentationConfig()
    transform = AugmentationFactory(aug_config).stress_test()

    dataset = ImageNetSubset(
        data_root, split="validation", transform=transform,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _smoke_test(data_root: Path = DEFAULT_DATA_ROOT) -> None:
    """Quick sanity check. Prints split sizes and inspects one train batch."""
    print(f"Loading dataloaders from: {data_root}\n")

    train_loader, dev_loader, val_loader = get_dataloaders(
        data_root=data_root,
        batch_size=16,
        num_workers=0,  # 0 keeps this script simple to debug.
    )

    print()
    print(f"  train      : {len(train_loader.dataset):>5} images, "
          f"{len(train_loader):>4} batches")
    print(f"  dev        : {len(dev_loader.dataset):>5} images, "
          f"{len(dev_loader):>4} batches")
    print(f"  validation : {len(val_loader.dataset):>5} images, "
          f"{len(val_loader):>4} batches")

    # Grab one train batch and look at it.
    images, labels = next(iter(train_loader))

    print("\nOne train batch:")
    print(f"  images shape : {tuple(images.shape)}")
    print(f"  images dtype : {images.dtype}")
    print(f"  labels shape : {tuple(labels.shape)}")
    print(f"  labels dtype : {labels.dtype}")
    print(f"  label min/max: {int(labels.min())} / {int(labels.max())}")

    # Labels MUST be in 0..19; anything else means the dataset is broken.
    assert 0 <= int(labels.min()) and int(labels.max()) <= 19, (
        f"labels out of range: min={int(labels.min())}, "
        f"max={int(labels.max())}"
    )
    print("\n[OK] labels are in the expected 0..19 range.")


if __name__ == "__main__":
    _smoke_test()
