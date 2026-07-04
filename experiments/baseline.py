"""Baseline training experiment for GRAD-SCOPE.

Trains an instrumented model on CIFAR-10 while a GradientLogger captures
per-layer gradient-flow dynamics. Serves as the reference run other
experiments are compared against.

After training, the run's full per-(epoch, layer) GSNR history is written to
``results/<run_name>_gsnr.csv`` (plus a ``*_metrics.json`` companion with the
validation-accuracy and training-loss curves), a per-layer GSNR summary table
is printed, and recommended ``tau_h`` / ``tau_n`` classification thresholds are
derived from the observed GSNR range.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from logger.gradient_logger import GradientLogger
from logger.layer_classifier import classify_all
from logger.lead_time import compute_all

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class BaselineConfig:
    """Hyperparameters and settings for the baseline run."""

    model: str = "resnet20"
    dataset: str = "cifar10"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 0
    log_every: int = 1
    results_dir: str = "results"
    figures_dir: str = "figures"
    data_dir: str = "data"
    num_workers: int = 2
    run_name: str = "baseline"
    # Pathology switches used by the failure-mode experiments.
    remove_bn: bool = False
    bad_init: bool = False
    # Optional caps on dataset size (for smoke tests); None = full dataset.
    train_subset: int | None = None
    val_subset: int | None = None


def get_device() -> torch.device:
    """Return the MPS device when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: BaselineConfig) -> nn.Module:
    """Instantiate the model named in the config."""
    if config.model == "resnet20":
        from models.resnet20 import get_resnet20

        model = get_resnet20(remove_bn=config.remove_bn)
    elif config.model == "vgg11":
        from models.vgg11 import get_vgg11

        model = get_vgg11(remove_bn=config.remove_bn)
    else:
        raise ValueError(f"unknown model {config.model!r} (expected 'resnet20' or 'vgg11')")

    if config.bad_init:
        model.bad_init()
    return model


def build_dataloaders(config: BaselineConfig) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 train/val loaders with standard augmentation and normalization."""
    if config.dataset != "cifar10":
        raise ValueError(f"unknown dataset {config.dataset!r} (expected 'cifar10')")

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=False, download=True, transform=val_transform
    )
    if config.train_subset is not None:
        train_set = Subset(train_set, range(min(config.train_subset, len(train_set))))
    if config.val_subset is not None:
        val_set = Subset(val_set, range(min(config.val_subset, len(val_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy of ``model`` over ``loader``."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += targets.size(0)
    return correct / max(total, 1)


def _layer_norms(logger: GradientLogger) -> dict[str, float]:
    """Per-layer gradient magnitude (RMS of grad elements) from the latest pass."""
    norms: dict[str, float] = {}
    for name, rec in logger.current.items():
        norms[name] = math.sqrt(rec["grad_mean"] ** 2 + max(rec["grad_var"], 0.0))
    return norms


def summarize_gsnr(logger: GradientLogger) -> dict[str, dict[str, float]]:
    """Per-layer min / max / mean GSNR across every epoch the logger observed."""
    per_layer: dict[str, list[float]] = {}
    for rec in logger.records:
        per_layer.setdefault(rec["layer"], []).append(rec["gsnr"])
    return {
        name: {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
        }
        for name, vals in sorted(per_layer.items())
    }


def recommend_thresholds(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """Derive tau_h / tau_n classification thresholds from the observed GSNR range.

    In a healthy run every layer's typical GSNR should classify as HEALTHY, so
    ``tau_h`` is set an order of magnitude below the 10th percentile of the
    per-layer mean GSNRs, and ``tau_n`` an order of magnitude below that. A
    layer must fall well outside the observed healthy band to be flagged.
    """
    means = [s["mean"] for s in summary.values() if math.isfinite(s["mean"]) and s["mean"] > 0]
    if not means:
        return {"tau_h": 0.1, "tau_n": 0.01}
    p10 = float(np.percentile(means, 10))
    tau_h = p10 / 10.0
    tau_n = tau_h / 10.0
    return {"tau_h": tau_h, "tau_n": tau_n}


def print_summary(summary: dict[str, dict[str, float]], thresholds: dict[str, float]) -> None:
    """Print the per-layer GSNR table and the recommended thresholds."""
    name_width = max([len(n) for n in summary] + [len("layer")])
    header = f"{'layer':<{name_width}}  {'min GSNR':>12}  {'max GSNR':>12}  {'mean GSNR':>12}"
    print("\nPer-layer GSNR summary (across all epochs):")
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        print(
            f"{name:<{name_width}}  {stats['min']:>12.4e}  "
            f"{stats['max']:>12.4e}  {stats['mean']:>12.4e}"
        )

    print("\nRecommended classification thresholds (from observed GSNR range):")
    print(f"  tau_h = {thresholds['tau_h']:.4e}   (GSNR >= tau_h -> HEALTHY)")
    print(f"  tau_n = {thresholds['tau_n']:.4e}   (GSNR <  tau_n -> NOISY)")


def run_baseline(config: BaselineConfig) -> dict:
    """Run the baseline training loop with gradient logging; return metrics."""
    torch.manual_seed(config.seed)
    device = get_device()
    print(f"[{config.run_name}] device: {device}")

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    logger = GradientLogger(model, results_dir=config.results_dir)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    val_accs: list[float] = []
    train_losses: list[float] = []
    layer_states_per_epoch: list[dict] = []
    layer_norms_per_epoch: list[dict[str, float]] = []

    start = time.time()
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * targets.size(0)
            seen += targets.size(0)

        train_loss = running_loss / max(seen, 1)
        val_acc = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        # Snapshot gradient health for this epoch, then flush live_state.json.
        layer_states_per_epoch.append(classify_all(logger))
        layer_norms_per_epoch.append(_layer_norms(logger))
        logger.step_epoch(val_accuracy=val_acc, train_loss=train_loss)

        if config.log_every and (epoch + 1) % config.log_every == 0:
            print(
                f"[{config.run_name}] epoch {epoch + 1:>3}/{config.epochs}  "
                f"train loss {train_loss:.4f}  val acc {val_acc:.4f}"
            )

    elapsed = time.time() - start
    logger.remove()

    # Persist the full GSNR history and the metric curves.
    csv_path = os.path.join(config.results_dir, f"{config.run_name}_gsnr.csv")
    logger.save_csv(csv_path)
    metrics_path = os.path.join(config.results_dir, f"{config.run_name}_gsnr_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"val_accs": val_accs, "train_losses": train_losses}, f, indent=2)
    print(f"\n[{config.run_name}] wrote {csv_path} and {metrics_path}")

    summary = summarize_gsnr(logger)
    thresholds = recommend_thresholds(summary)
    print_summary(summary, thresholds)

    lead_times = compute_all(
        val_accs, layer_states_per_epoch, layer_norms_per_epoch, train_losses
    )

    return {
        "run_name": config.run_name,
        "device": str(device),
        "epochs": config.epochs,
        "elapsed_seconds": elapsed,
        "final_val_acc": val_accs[-1] if val_accs else None,
        "best_val_acc": max(val_accs) if val_accs else None,
        "val_accs": val_accs,
        "train_losses": train_losses,
        "gsnr_summary": summary,
        "recommended_thresholds": thresholds,
        "lead_times": lead_times,
        "csv_path": csv_path,
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":
    metrics = run_baseline(BaselineConfig())
    print(
        f"\n[baseline] done in {metrics['elapsed_seconds']:.0f}s — "
        f"final val acc {metrics['final_val_acc']:.4f}, "
        f"best val acc {metrics['best_val_acc']:.4f}"
    )
