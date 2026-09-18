#!/usr/bin/env python3
"""
Training script for my_team.

This script:
  1. Loads train/dev DataLoaders from data_utils.get_dataloaders.
  2. Builds the model defined in model.py (from scratch, no pretrained).
  3. Trains with CrossEntropyLoss + Adam.
  4. Prints train loss and dev accuracy after each epoch.
  5. Keeps a CPU snapshot of the BEST model (highest clean dev accuracy)
     and saves it IMMEDIATELY to weights.joblib whenever it improves,
     using an atomic write so a crash mid-save cannot corrupt the file.
     Optionally also writes a timestamped history to --checkpoint-dir.
  6. Supports optional --patience for early stopping on no-improvement.

Run from anywhere:
    python submissions/my_team/train.py
or:
    cd submissions/my_team && python train.py

Quick smoke test (only a handful of batches, useful on CPU):
    python submissions/my_team/train.py --smoke-test

Override individual hyper-parameters:
    python submissions/my_team/train.py --epochs 30 --batch-size 128 --lr 1e-3

Point at a different data folder:
    python submissions/my_team/train.py --data-root /content/dataset_split

Use the standard augmentation preset and also report stress-test accuracy:
    python submissions/my_team/train.py --augment standard --eval-stress

Continue training from the current best weights:
    python submissions/my_team/train.py --augment standard --resume --lr 1e-4
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.optim import Adam

# Make project root importable so we can use data_utils and base_model
# no matter what the current working directory is.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_augmentation import AugmentationConfig, AugmentationFactory  # noqa: E402
from data_utils import get_dataloaders, get_stress_test_loader  # noqa: E402
from model import ModelArchitecture  # noqa: E402


# ── settings (edit these freely) ──────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "dataset_split"
OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
NUM_CLASSES = 20

# Smoke-test settings (only used when --smoke-test is passed).
SMOKE_EPOCHS = 1
SMOKE_TRAIN_BATCHES = 3
SMOKE_DEV_BATCHES = 3
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train my_team's Small ResNet.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Quickly run a few train+eval batches and save weights. "
            "Useful for verifying the pipeline on CPU. "
            "Overrides --epochs and --num-workers."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for both train and dev loaders (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for Adam (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"DataLoader worker processes (default: {NUM_WORKERS}).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=(
            "Folder containing train/ dev/ validation/ subfolders "
            f"(default: {DATA_ROOT})."
        ),
    )
    parser.add_argument(
        "--augment",
        choices=["none", "standard", "strong"],
        default="none",
        help=(
            "Train-time augmentation preset. "
            "'none' = basic torchvision train transform (default). "
            "'standard' = AugmentationFactory.standard_training(). "
            "'strong'   = AugmentationFactory.strong_training(). "
            "For 'standard' and 'strong' the slow heuristic-mask "
            "background augmentations are disabled by default."
        ),
    )
    parser.add_argument(
        "--eval-stress",
        action="store_true",
        help=(
            "After training, also report stress-test accuracy on the "
            "validation split (AugmentationFactory.stress_test())."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue training from the default weights.joblib instead of "
            "starting from random initialization."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Continue training from a specific weights.joblib file. "
            "Cannot be combined with --resume."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional folder to save best-weights checkpoints into, "
            "in addition to the main weights.joblib. Two files are "
            "written there on every improvement: best_weights.joblib "
            "(latest) and best_weights_epoch_{N}_acc_{A}.joblib "
            "(history). The folder is created if missing."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "If set, stop training after this many consecutive epochs "
            "with no clean dev_acc improvement. Best weights are still "
            "saved (they are already saved on each improvement)."
        ),
    )
    args = parser.parse_args()
    if args.resume and args.resume_from is not None:
        parser.error("--resume and --resume-from cannot be used together")
    return args


def atomic_joblib_dump(obj, path: Path) -> None:
    """
    Dump `obj` to `path` atomically.

    We write to `<path>.tmp` first and only then rename it onto `path`.
    On POSIX filesystems os.replace is atomic, so a process that is
    killed mid-write can NEVER leave us with a half-written
    weights.joblib (it would still have the previous good version).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp_path)
    tmp_path.replace(path)


def snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU-resident, detached, cloned copy of model.state_dict()."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def load_weights_into_model(model: nn.Module, weights_path: Path) -> None:
    """Load a joblib-saved state_dict into `model`."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Resume weights not found: {weights_path}")

    state_dict = joblib.load(weights_path)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected {weights_path} to contain a state_dict dict, "
            f"got {type(state_dict).__name__}"
        )

    model.load_state_dict(state_dict)


def save_best_checkpoint(
    state_dict: dict[str, torch.Tensor],
    output: Path,
    checkpoint_dir: Path | None,
    epoch: int,
    dev_acc: float,
) -> list[Path]:
    """
    Save `state_dict` to `output` atomically, and (optionally) also write
    two history files into `checkpoint_dir`.

    Returns the list of paths that were written, so the caller can print
    them.
    """
    paths_written: list[Path] = []

    # The main artifact the grader will read.
    atomic_joblib_dump(state_dict, output)
    paths_written.append(output)

    # Optional history folder for crash safety.
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        latest_path = checkpoint_dir / "best_weights.joblib"
        epoch_path = checkpoint_dir / (
            f"best_weights_epoch_{epoch}_acc_{dev_acc:.4f}.joblib"
        )

        atomic_joblib_dump(state_dict, latest_path)
        # epoch_path is unique per call, so a plain dump is fine.
        joblib.dump(state_dict, epoch_path)

        paths_written.extend([latest_path, epoch_path])

    return paths_written


def build_train_transform_for(augment_mode: str):
    """
    Resolve --augment into the train transform to pass to get_dataloaders.

    Returns None for 'none', which makes get_dataloaders fall back to its
    own basic train transform (no behavior change vs. before --augment
    existed).

    For 'standard' and 'strong' we override the heuristic-mask background
    augmentations to 0.0 so the slow Python pixel loop never runs.
    """
    if augment_mode == "none":
        return None

    aug_config = replace(
        AugmentationConfig(),

        background_delete_prob=0.03,
        background_replace_prob=0.05,
        strong_background_delete_prob=0.08,
        strong_background_replace_prob=0.12,

        perspective_prob=0.05,
        noise_prob=0.10,
        random_erasing_prob=0.10,
        posterize_prob=0.0,
        grayscale_prob=0.02,
    )
    factory = AugmentationFactory(aug_config)

    if augment_mode == "standard":
        return factory.standard_training()
    if augment_mode == "strong":
        return factory.strong_training()
    raise ValueError(f"Unknown --augment value: {augment_mode}")


def pick_device() -> torch.device:
    """Use CUDA if available, otherwise plain CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Run one pass over the train loader. Returns mean loss per sample.

    If max_batches is set, stop after that many batches (used by --smoke-test).
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Compute accuracy on a loader (no shuffle, no augmentation).

    If max_batches is set, stop after that many batches (used by --smoke-test).
    """
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def main() -> None:
    args = parse_args()

    device = pick_device()

    # Resolve the effective settings. In normal mode they come straight
    # from the CLI (or its defaults). In smoke-test mode we override the
    # ones that would make the smoke test slow, so --smoke-test behaves
    # exactly like before regardless of the other flags.
    if args.smoke_test:
        num_epochs = SMOKE_EPOCHS
        num_workers = 0  # avoid multi-process startup cost for 3 batches
        max_train_batches = SMOKE_TRAIN_BATCHES
        max_dev_batches = SMOKE_DEV_BATCHES
    else:
        num_epochs = args.epochs
        num_workers = args.num_workers
        max_train_batches = None
        max_dev_batches = None

    batch_size = args.batch_size
    learning_rate = args.lr
    data_root = args.data_root

    # Print the final, resolved settings so the run is self-documenting.
    print("Settings:")
    print(f"  device         : {device}")
    print(f"  data_root      : {data_root}")
    print(f"  epochs         : {num_epochs}")
    print(f"  batch_size     : {batch_size}")
    print(f"  lr             : {learning_rate}")
    print(f"  num_workers    : {num_workers}")
    print(f"  augment        : {args.augment}")
    if args.resume:
        print(f"  resume_from    : {OUTPUT}")
    else:
        print(f"  resume_from    : {args.resume_from}")
    print(f"  eval_stress    : {args.eval_stress}")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  patience       : {args.patience}")
    if args.smoke_test:
        print("  mode           : smoke-test (3 train + 3 dev batches, 1 epoch)")
    print()

    # 1. Data.
    train_transform = build_train_transform_for(args.augment)
    train_loader, dev_loader, _val_loader = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
    )

    # 2. Model, loss, optimizer.
    model = ModelArchitecture(num_classes=NUM_CLASSES).to(device)
    resume_path = OUTPUT if args.resume else args.resume_from
    if resume_path is not None:
        load_weights_into_model(model, resume_path)
        print(f"Loaded resume weights from: {resume_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters\n")

    # 3. Training loop.
    #    We keep a CPU snapshot of the state_dict from the epoch with the
    #    highest clean dev accuracy, AND we save it to disk immediately
    #    whenever it improves. That way an unexpected disconnect (Colab,
    #    SIGINT, etc.) can never lose more than the most recent epoch.
    best_dev_acc = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    dev_acc = 0.0  # so the "final epoch" print never sees an undefined value
    epochs_since_improvement = 0
    stopped_early = False

    if resume_path is not None:
        print("Evaluating loaded baseline before augmented fine-tuning...")
        best_dev_acc = evaluate_accuracy(
            model, dev_loader, device,
            max_batches=max_dev_batches,
        )
        best_state_dict = snapshot_state_dict(model)
        print(f"Loaded baseline dev_acc       : {best_dev_acc:.4f}")

    for epoch in range(1, num_epochs + 1):
        if args.smoke_test:
            print(f"Smoke test mode: training only {SMOKE_TRAIN_BATCHES} batches")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_batches=max_train_batches,
        )

        if args.smoke_test:
            print(f"Smoke test mode: evaluating only {SMOKE_DEV_BATCHES} dev batches")
        dev_acc = evaluate_accuracy(
            model, dev_loader, device,
            max_batches=max_dev_batches,
        )

        # "Strictly greater" so the earliest epoch reaching a given
        # accuracy wins ties (smaller model, less risk of late overfit).
        improved = dev_acc > best_dev_acc
        if improved:
            best_dev_acc = dev_acc
            best_epoch = epoch
            epochs_since_improvement = 0
            best_state_dict = snapshot_state_dict(model)
        else:
            epochs_since_improvement += 1

        marker = "  *" if improved else ""
        print(
            f"epoch {epoch:>2}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"dev_acc={dev_acc:.4f}{marker}"
        )

        # Save IMMEDIATELY on improvement, before the next epoch can
        # crash. This is the whole point of this script: never lose a
        # good model.
        if improved:
            paths = save_best_checkpoint(
                state_dict=best_state_dict,
                output=OUTPUT,
                checkpoint_dir=args.checkpoint_dir,
                epoch=epoch,
                dev_acc=dev_acc,
            )
            print(
                f"  Saved new best weights to {paths[0]} "
                f"at epoch {epoch} with dev_acc={dev_acc:.4f}"
            )
            for extra in paths[1:]:
                print(f"  Also saved to {extra}")

        # Optional early stopping.
        if args.patience is not None and epochs_since_improvement >= args.patience:
            print(
                f"\nEarly stopping: no improvement for "
                f"{epochs_since_improvement} consecutive epochs "
                f"(patience={args.patience})."
            )
            stopped_early = True
            break

    # Safety net: if for some reason no improvement was ever recorded
    # (e.g. num_epochs == 0), fall back to the current model's weights.
    if best_state_dict is None:
        best_state_dict = snapshot_state_dict(model)
        # And make sure weights.joblib exists on disk for the grader.
        atomic_joblib_dump(best_state_dict, OUTPUT)

    print(f"\nClean dev accuracy (final epoch): {dev_acc:.4f}")
    print(f"Best clean dev accuracy         : {best_dev_acc:.4f} "
          f"(at epoch {best_epoch})")
    if stopped_early:
        print(f"Training stopped early after epoch {epoch}.")

    # 4. Optional stress-test evaluation. Loaded with the BEST weights so
    #    the reported stress accuracy matches the model we will actually
    #    ship in weights.joblib. We also turn off the slow heuristic-mask
    #    background augmentations on this loader.
    if args.eval_stress:
        print("\nRunning stress-test evaluation on validation split "
              f"(using best epoch {best_epoch})...")
        # Load best weights into the model in place. State_dict is on
        # CPU; PyTorch handles the cross-device copy.
        model.load_state_dict(best_state_dict)

        stress_config = replace(
            AugmentationConfig(),
            background_delete_prob=0.0,
            background_replace_prob=0.0,
            stress_background_replace_prob=0.0,
        )
        stress_loader = get_stress_test_loader(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            config=stress_config,
        )
        stress_acc = evaluate_accuracy(
            model, stress_loader, device,
            max_batches=max_dev_batches,  # cap in smoke-test mode too
        )
        print(f"Stress-test accuracy on validation: {stress_acc:.4f}")

    # 5. Final "make sure" save. weights.joblib was already kept up to
    #    date on every improvement, but we re-dump it once more here so
    #    a script that wraps train.py can rely on the file being present
    #    and current at exit time (belt-and-suspenders).
    atomic_joblib_dump(best_state_dict, OUTPUT)
    print(f"\nFinal weights.joblib reflects epoch {best_epoch} "
          f"(dev_acc={best_dev_acc:.4f}).")
    print(f"Best weights location: {OUTPUT}")
    if args.checkpoint_dir is not None:
        print(f"Checkpoint history   : {Path(args.checkpoint_dir).resolve()}")


if __name__ == "__main__":
    main()
