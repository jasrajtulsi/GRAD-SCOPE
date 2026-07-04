"""VGG-11 (CIFAR-style) for GRAD-SCOPE gradient-flow experiments.

Standard VGG-11 ("A") convolutional stack adapted for 32x32 CIFAR-10 inputs:
adaptive average pooling before a compact classifier head. Supports building the
network without BatchNorm (``remove_bn=True``) to compare gradient-flow dynamics
against the batch-normalized baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Standard VGG-11 configuration ("A"): numbers are conv channels, "M" is maxpool.
VGG11_CONFIG: list[int | str] = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11(nn.Module):
    """VGG-11 convolutional network for 32x32 inputs."""

    def __init__(self, num_classes: int = 10, remove_bn: bool = False) -> None:
        super().__init__()
        self.remove_bn = remove_bn
        self.features = self._make_features(remove_bn)
        # Adaptive pooling makes the classifier input size independent of the
        # spatial resolution (7x7 keeps the classic VGG head shape).
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # ReLUs are deliberately not in-place: GradientLogger attaches full
        # backward hooks to every leaf module, and autograd forbids in-place
        # modification of a hooked module's output.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def _make_features(self, remove_bn: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for v in VGG11_CONFIG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            out_channels = int(v)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if not remove_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def bad_init(self) -> None:
        """Pathological initialization: set every weight/bias to the constant 0.001.

        Used to induce degenerate gradient-flow dynamics for GRAD-SCOPE.
        """
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(0.001)


def get_vgg11(remove_bn: bool = False, num_classes: int = 10) -> VGG11:
    """Build a VGG-11. When ``remove_bn`` is True, no BatchNorm layers are added
    (the conv stack is plain conv -> ReLU)."""
    return VGG11(num_classes=num_classes, remove_bn=remove_bn)


if __name__ == "__main__":
    model = get_vgg11()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input shape:      {tuple(x.shape)}")
    print(f"output shape:     {tuple(y.shape)}")
    print(f"parameter count:  {n_params:,}")
