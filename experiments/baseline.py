"""Baseline training experiment for GRAD-SCOPE.

Trains an instrumented model on CIFAR-10 while a GradientLogger captures
per-layer gradient-flow dynamics. Serves as the reference run other
experiments are compared against.

Stub only — no implementation yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from logger import GradientLogger, LeadTimeAnalyzer


@dataclass
class BaselineConfig:
    """Hyperparameters and settings for the baseline run."""

    model: str = "resnet20"
    dataset: str = "cifar10"
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 0
    log_every: int = 1
    results_dir: str = "results"
    figures_dir: str = "figures"


def build_model(config: BaselineConfig) -> nn.Module:
    """Instantiate the model named in the config."""
    raise NotImplementedError


def run_baseline(config: BaselineConfig) -> dict:
    """Run the baseline training loop with gradient logging; return metrics."""
    raise NotImplementedError
