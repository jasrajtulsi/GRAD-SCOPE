"""Per-layer gradient-flow logger for GRAD-SCOPE.

Registers a full backward hook on every leaf module of a model and, after each
backward pass, records the gradient mean, variance, and GSNR (gradient
signal-to-noise ratio) flowing through that module. After every epoch the
current state is also written to ``results/live_state.json`` — the snapshot the
MCP diagnostic server reads to triage a training run in flight.

GSNR is defined here as ``mean**2 / (var + 1e-8)`` over the elements of the
gradient tensor arriving at a module's output (``grad_output[0]``): a high value
means a consistent, low-noise gradient signal; a value near zero means the
gradient is dominated by noise or has vanished.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Callable

import torch
import torch.nn as nn


class GradientLogger:
    """Capture per-layer gradient statistics over the course of training."""

    def __init__(
        self,
        model: nn.Module,
        results_dir: str = "results",
        live_state_path: str | None = None,
    ) -> None:
        self.model = model
        self.results_dir = results_dir
        self.live_state_path = live_state_path or os.path.join(results_dir, "live_state.json")

        self.epoch: int = 0
        # One record per (epoch, layer) backward observation.
        self.records: list[dict] = []
        # Most recent record per layer, refreshed on every backward pass.
        self.current: dict[str, dict] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        self._attach()

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def _is_leaf(module: nn.Module) -> bool:
        return next(module.children(), None) is None

    def _attach(self) -> None:
        """Register a full backward hook on every leaf module."""
        for name, module in self.model.named_modules():
            if name == "" or not self._is_leaf(module):
                continue
            handle = module.register_full_backward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, grad_input, grad_output) -> None:
            if not grad_output or grad_output[0] is None:
                return
            grad = grad_output[0].detach()
            mean = float(grad.mean().item())
            var = float(grad.var(unbiased=False).item())
            gsnr = (mean * mean) / (var + 1e-8)
            record = {
                "epoch": self.epoch,
                "layer": name,
                "grad_mean": mean,
                "grad_var": var,
                "gsnr": gsnr,
                "state": self._classify_state(mean, var),
            }
            self.records.append(record)
            self.current[name] = record

        return hook

    @staticmethod
    def _classify_state(mean: float, var: float) -> str:
        """Heuristically label a layer's gradient health for the live snapshot."""
        magnitude = abs(mean) + math.sqrt(max(var, 0.0))
        if not math.isfinite(magnitude):
            return "exploding"
        if magnitude < 1e-7:
            return "vanishing"
        if magnitude > 1e3:
            return "exploding"
        return "healthy"

    # ---------------------------------------------------------------- epochs
    def step_epoch(
        self,
        val_accuracy: float | None = None,
        train_loss: float | None = None,
    ) -> None:
        """Flush the current epoch's state to ``live_state.json`` and advance.

        Call once at the end of each epoch, after that epoch's backward passes.
        """
        self._write_live_state(val_accuracy=val_accuracy, train_loss=train_loss)
        self.epoch += 1

    def _write_live_state(
        self,
        val_accuracy: float | None,
        train_loss: float | None,
    ) -> None:
        os.makedirs(os.path.dirname(self.live_state_path) or ".", exist_ok=True)
        state = {
            "epoch": self.epoch,
            "val_accuracy": val_accuracy,
            "training_loss": train_loss,
            "timestamp": time.time(),
            "current_gsnr": self.get_current_gsnr(),
            "layer_states": {
                name: {
                    "grad_mean": rec["grad_mean"],
                    "grad_var": rec["grad_var"],
                    "gsnr": rec["gsnr"],
                    "state": rec["state"],
                }
                for name, rec in self.current.items()
            },
        }
        tmp_path = self.live_state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.live_state_path)  # atomic for a concurrent reader

    # --------------------------------------------------------------- queries
    def get_current_gsnr(self) -> dict[str, float]:
        """Return the latest GSNR per layer (from the most recent backward pass)."""
        return {name: rec["gsnr"] for name, rec in self.current.items()}

    def get_gsnr_history(self, layer_name: str) -> list[tuple[int, float]]:
        """Return the (epoch, GSNR) history for a single layer, in order."""
        return [
            (rec["epoch"], rec["gsnr"])
            for rec in self.records
            if rec["layer"] == layer_name
        ]

    # ------------------------------------------------------------ lifecycle
    def save_csv(self, path: str) -> None:
        """Write all collected per-(epoch, layer) records to a CSV file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = ["epoch", "layer", "grad_mean", "grad_var", "gsnr", "state"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    def remove(self) -> None:
        """Remove all registered backward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


if __name__ == "__main__":
    torch.manual_seed(0)

    # Tiny 3-layer model: Linear -> ReLU -> Linear.
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    logger = GradientLogger(model, results_dir="results")

    x = torch.randn(8, 10)
    target = torch.randint(0, 2, (8,))

    for _ in range(3):
        out = model(x)
        loss = nn.functional.cross_entropy(out, target)
        model.zero_grad()
        loss.backward()
        logger.step_epoch(val_accuracy=0.5, train_loss=float(loss.item()))

    current = logger.get_current_gsnr()
    print("hooked layers:", list(current.keys()))
    for name, gsnr in current.items():
        print(f"  {name:>2}  GSNR = {gsnr:.6g}")

    assert current, "no GSNR values were recorded"
    assert all(math.isfinite(v) for v in current.values()), "found NaN/inf GSNR"
    print("all GSNR values are finite (no NaN) ✓")

    logger.save_csv("results/gsnr_test.csv")
    print("wrote results/gsnr_test.csv and results/live_state.json")
    logger.remove()
