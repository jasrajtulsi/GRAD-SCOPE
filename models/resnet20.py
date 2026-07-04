"""ResNet-20 (CIFAR-style) for GRAD-SCOPE gradient-flow experiments.

3 groups of 3 residual blocks with 16 / 32 / 64 filters. Supports building the
network without BatchNorm (``remove_bn=True``) so its gradient-flow dynamics can
be compared against the batch-normalized baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(num_features: int, remove_bn: bool) -> nn.Module:
    """Return a BatchNorm2d, or nn.Identity when BatchNorm is disabled."""
    return nn.Identity() if remove_bn else nn.BatchNorm2d(num_features)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual (identity/projection) shortcut."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        remove_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = _norm(out_channels, remove_bn)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = _norm(out_channels, remove_bn)

        # Projection shortcut when the shape changes (stride or channel count).
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _norm(out_channels * self.expansion, remove_bn),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """20-layer residual network for 32x32 inputs (3 stages x 3 blocks)."""

    def __init__(self, num_classes: int = 10, remove_bn: bool = False) -> None:
        super().__init__()
        self.remove_bn = remove_bn
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = _norm(16, remove_bn)

        self.layer1 = self._make_stage(16, num_blocks=3, stride=1)
        self.layer2 = self._make_stage(32, num_blocks=3, stride=2)
        self.layer3 = self._make_stage(64, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_channels, out_channels, s, self.remove_bn))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

    def bad_init(self) -> None:
        """Pathological initialization: set every weight/bias to the constant 0.001.

        Used to induce degenerate gradient-flow dynamics for GRAD-SCOPE.
        """
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(0.001)


def get_resnet20(remove_bn: bool = False, num_classes: int = 10) -> ResNet20:
    """Build a ResNet-20. When ``remove_bn`` is True, every BatchNorm layer is
    replaced with ``nn.Identity()``."""
    return ResNet20(num_classes=num_classes, remove_bn=remove_bn)


if __name__ == "__main__":
    model = get_resnet20()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input shape:      {tuple(x.shape)}")
    print(f"output shape:     {tuple(y.shape)}")
    print(f"parameter count:  {n_params:,}")
