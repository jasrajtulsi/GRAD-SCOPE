"""Layer classification for GRAD-SCOPE.

Maps modules in a network to categories (conv, batchnorm, linear, shortcut,
etc.) and to a normalized depth, so gradient statistics can be grouped and
compared across architectures.

Stub only — no implementation yet.
"""

from __future__ import annotations

from enum import Enum

import torch.nn as nn


class LayerType(Enum):
    """Coarse category a module is assigned to for gradient grouping."""

    CONV = "conv"
    LINEAR = "linear"
    NORM = "norm"
    SHORTCUT = "shortcut"
    OTHER = "other"


class LayerClassifier:
    """Assign a LayerType and a normalized depth to each parameterized module."""

    def __init__(self, model: nn.Module) -> None:
        raise NotImplementedError

    def classify(self, name: str, module: nn.Module) -> LayerType:
        """Return the LayerType for a single named module."""
        raise NotImplementedError

    def depth_of(self, name: str) -> float:
        """Return the module's depth normalized to [0, 1] (input -> output)."""
        raise NotImplementedError

    def layer_map(self) -> dict[str, LayerType]:
        """Return a mapping from module name to its LayerType."""
        raise NotImplementedError
