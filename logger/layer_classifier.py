"""Layer classification for GRAD-SCOPE.

Two distinct notions of "classification" live here:

1. :class:`LayerClassifier` / :class:`LayerType` map modules to *architectural*
   categories (conv, batchnorm, linear, shortcut, ...) and to a normalized
   depth, so gradient statistics can be grouped across architectures.

2. :func:`classify` and friends map a layer's *gradient-flow health* to one of
   :class:`GradientState` (HEALTHY, STAGNANT, NOISY, DEAD) from its GSNR history.
   This is what the diagnostic layer and the state heatmap consume.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence, Union

import torch.nn as nn


class LayerType(Enum):
    """Coarse category a module is assigned to for gradient grouping."""

    CONV = "conv"
    LINEAR = "linear"
    NORM = "norm"
    SHORTCUT = "shortcut"
    OTHER = "other"


_NORM_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


class LayerClassifier:
    """Assign a LayerType and a normalized depth to each parameterized module.

    On construction, walks the model's leaf modules in ``named_modules`` order
    (which follows the forward-pass definition order) and records every module
    that owns at least one parameter. Depth is that ordering normalized to
    [0, 1], so statistics can be compared across architectures of different
    sizes.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._order: list[str] = []
        self._types: dict[str, LayerType] = {}
        for name, module in model.named_modules():
            if name == "" or next(module.children(), None) is not None:
                continue  # composite module; only leaves are classified
            if next(module.parameters(recurse=False), None) is None:
                continue  # no parameters (ReLU, Identity, pooling, ...)
            self._order.append(name)
            self._types[name] = self.classify(name, module)

    def classify(self, name: str, module: nn.Module) -> LayerType:
        """Return the LayerType for a single named module."""
        # A projection inside a residual shortcut is its own category, whatever
        # its module type, so shortcut gradients can be analyzed separately.
        if "shortcut" in name or "downsample" in name:
            return LayerType.SHORTCUT
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return LayerType.CONV
        if isinstance(module, _NORM_MODULES):
            return LayerType.NORM
        if isinstance(module, nn.Linear):
            return LayerType.LINEAR
        return LayerType.OTHER

    def depth_of(self, name: str) -> float:
        """Return the module's depth normalized to [0, 1] (input -> output)."""
        if name not in self._types:
            raise KeyError(f"unknown or unparameterized module: {name!r}")
        if len(self._order) == 1:
            return 0.0
        return self._order.index(name) / (len(self._order) - 1)

    def layer_map(self) -> dict[str, LayerType]:
        """Return a mapping from module name to its LayerType."""
        return dict(self._types)


# --------------------------------------------------------------------------- #
# Gradient-flow health classification                                         #
# --------------------------------------------------------------------------- #

# Thresholds below which a gradient is considered effectively gone.
_DEAD_FLOOR = 1e-10   # per-epoch GSNR counted as "no signal" for the DEAD run
_STAGNANT_FLOOR = 1e-8  # current GSNR below this: gradient has all but vanished


class GradientState(str, Enum):
    """Health of the gradient signal flowing through a layer.

    ``str`` subclass so members compare equal to their value and serialize
    cleanly to JSON / CSV / plot labels.
    """

    HEALTHY = "HEALTHY"      # strong, informative gradient
    STAGNANT = "STAGNANT"    # gradient has essentially vanished (learning stalled)
    NOISY = "NOISY"          # signal present but weak / noise-dominated
    DEAD = "DEAD"            # flatlined near zero for a sustained run of epochs


# Ordered from most to least severe. Handy for assigning colours in a heatmap.
STATE_ORDER: list[GradientState] = [
    GradientState.DEAD,
    GradientState.STAGNANT,
    GradientState.NOISY,
    GradientState.HEALTHY,
]


def _gsnr_values(gsnr_history: Sequence[Union[float, tuple]]) -> list[float]:
    """Extract the numeric GSNR values from a history.

    Accepts either a flat list of floats or a list of ``(epoch, gsnr)`` pairs
    (the form :meth:`GradientLogger.get_gsnr_history` returns).
    """
    values: list[float] = []
    for item in gsnr_history:
        if isinstance(item, (tuple, list)):
            values.append(float(item[-1]))
        else:
            values.append(float(item))
    return values


def classify(
    gsnr_history: Sequence[Union[float, tuple]],
    patience: int = 5,
    tau_h: float = 0.1,
    tau_n: float = 0.01,
) -> GradientState:
    """Classify a layer's gradient health from its GSNR history.

    ``gsnr_history`` is the ordered GSNR readings for one layer (floats, or
    ``(epoch, gsnr)`` pairs). The most recent reading is the "current" GSNR.

    Rules, applied in order:

    * **DEAD** — the last ``patience`` readings are *all* below ``1e-10``. This
      requires a sustained run: a history shorter than ``patience`` can never be
      DEAD (it hasn't had time to prove a flatline).
    * **STAGNANT** — current GSNR below ``1e-8`` (gradient all but vanished).
    * **NOISY** — current GSNR below ``tau_n``.
    * **HEALTHY** — current GSNR at or above ``tau_h``.
    * Otherwise (the hysteresis band ``tau_n <= current < tau_h``) the signal is
      not yet strong enough to call healthy, so it is reported **NOISY**.

    An empty history is treated as DEAD (no signal at all).
    """
    values = _gsnr_values(gsnr_history)
    if not values:
        return GradientState.DEAD

    current = values[-1]

    # DEAD: a sustained flatline over the full patience window.
    window = values[-patience:]
    if len(window) >= patience and all(v < _DEAD_FLOOR for v in window):
        return GradientState.DEAD

    if current < _STAGNANT_FLOOR:
        return GradientState.STAGNANT
    if current < tau_n:
        return GradientState.NOISY
    if current >= tau_h:
        return GradientState.HEALTHY
    # tau_n <= current < tau_h: present but not strong enough to be healthy.
    return GradientState.NOISY


def classify_all(
    logger,
    patience: int = 5,
    tau_h: float = 0.1,
    tau_n: float = 0.01,
) -> dict[str, GradientState]:
    """Classify every layer a :class:`GradientLogger` has observed.

    Returns a mapping ``layer_name -> GradientState`` using each layer's full
    GSNR history up to the current epoch.
    """
    states: dict[str, GradientState] = {}
    for name in sorted(logger.current):
        history = logger.get_gsnr_history(name)
        states[name] = classify(
            history, patience=patience, tau_h=tau_h, tau_n=tau_n
        )
    return states


def get_state_heatmap_data(
    logger,
    patience: int = 5,
    tau_h: float = 0.1,
    tau_n: float = 0.01,
) -> dict:
    """Build a layer-by-epoch grid of gradient states for a heatmap plot.

    Returns a dict with:

    * ``"layers"`` — sorted layer names (rows).
    * ``"epochs"`` — sorted epoch indices (columns).
    * ``"states"`` — a 2-D list ``states[i][j]`` giving the state (as a string)
      of layer ``i`` as of epoch ``j``, classified from that layer's history up
      to and including epoch ``j``. Cells with no reading yet are ``"DEAD"``.

    State strings match :class:`GradientState` values; use :data:`STATE_ORDER`
    to map them to colour indices.
    """
    layers = sorted({r["layer"] for r in logger.records})
    epochs = sorted({r["epoch"] for r in logger.records})

    # Ordered (epoch, gsnr) history per layer.
    history: dict[str, list[tuple[int, float]]] = {name: [] for name in layers}
    for r in logger.records:
        history[r["layer"]].append((r["epoch"], r["gsnr"]))
    for name in layers:
        history[name].sort(key=lambda t: t[0])

    states: list[list[str]] = []
    for name in layers:
        row: list[str] = []
        for e in epochs:
            prefix = [pair for pair in history[name] if pair[0] <= e]
            state = classify(
                prefix, patience=patience, tau_h=tau_h, tau_n=tau_n
            )
            row.append(state.value)
        states.append(row)

    return {"layers": layers, "epochs": epochs, "states": states}


if __name__ == "__main__":
    # -- classify() on fake GSNR histories, covering every state ------------- #
    cases = [
        ("sustained flatline -> DEAD", [1e-12] * 6, GradientState.DEAD),
        ("short flatline (< patience) -> STAGNANT", [1e-12] * 3, GradientState.STAGNANT),
        ("recovered then vanished -> STAGNANT", [1.0, 1.0, 1e-9], GradientState.STAGNANT),
        ("weak signal below tau_n -> NOISY", [0.5, 0.2, 0.005], GradientState.NOISY),
        ("hysteresis band [tau_n, tau_h) -> NOISY", [0.05], GradientState.NOISY),
        ("strong signal above tau_h -> HEALTHY", [0.3, 0.4, 0.5], GradientState.HEALTHY),
        ("empty history -> DEAD", [], GradientState.DEAD),
        ("(epoch, gsnr) pairs accepted", [(0, 0.4), (1, 0.6)], GradientState.HEALTHY),
    ]
    print("classify() cases:")
    for label, history, expected in cases:
        got = classify(history)
        flag = "ok" if got == expected else "FAIL"
        print(f"  [{flag}] {label}: {got.value}")
        assert got == expected, f"{label}: expected {expected}, got {got}"

    # -- classify_all() / get_state_heatmap_data() on a fake logger --------- #
    class _FakeLogger:
        """Minimal stand-in exposing the attributes the two helpers read."""

        def __init__(self, series: dict[str, list[float]]) -> None:
            self.records = []
            self.current = {}
            for layer, gsnrs in series.items():
                for epoch, g in enumerate(gsnrs):
                    rec = {"epoch": epoch, "layer": layer, "gsnr": g}
                    self.records.append(rec)
                    self.current[layer] = rec

        def get_gsnr_history(self, layer_name):
            return [
                (r["epoch"], r["gsnr"])
                for r in self.records
                if r["layer"] == layer_name
            ]

    fake = _FakeLogger(
        {
            "layer0.healthy": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "layer1.noisy": [0.2, 0.1, 0.05, 0.02, 0.008, 0.006],
            "layer2.dead": [1e-12] * 6,
        }
    )

    print("\nclassify_all():")
    states = classify_all(fake)
    for name, state in states.items():
        print(f"  {name}: {state.value}")
    assert states["layer0.healthy"] == GradientState.HEALTHY
    assert states["layer1.noisy"] == GradientState.NOISY
    assert states["layer2.dead"] == GradientState.DEAD

    print("\nget_state_heatmap_data():")
    heat = get_state_heatmap_data(fake)
    print("  layers:", heat["layers"])
    print("  epochs:", heat["epochs"])
    for name, row in zip(heat["layers"], heat["states"]):
        print(f"    {name:>16}: {row}")
    assert heat["layers"] == ["layer0.healthy", "layer1.noisy", "layer2.dead"]
    assert heat["epochs"] == [0, 1, 2, 3, 4, 5]
    # The dead layer only crosses the patience threshold once 5 readings exist.
    assert heat["states"][2][-1] == GradientState.DEAD.value
    assert heat["states"][0][-1] == GradientState.HEALTHY.value

    # -- LayerClassifier on a small mixed model ------------------------------ #
    demo = nn.Sequential()
    demo.add_module("conv", nn.Conv2d(3, 8, 3, padding=1))
    demo.add_module("bn", nn.BatchNorm2d(8))
    demo.add_module("relu", nn.ReLU())  # unparameterized: excluded
    demo.add_module(
        "shortcut", nn.Sequential(nn.Conv2d(8, 8, 1))  # nested leaf -> shortcut.0
    )
    demo.add_module("fc", nn.Linear(8, 2))

    lc = LayerClassifier(demo)
    lmap = lc.layer_map()
    print("\nLayerClassifier.layer_map():")
    for name, ltype in lmap.items():
        print(f"  {name:>12}: {ltype.value}  (depth {lc.depth_of(name):.2f})")
    assert lmap["conv"] == LayerType.CONV
    assert lmap["bn"] == LayerType.NORM
    assert lmap["shortcut.0"] == LayerType.SHORTCUT
    assert lmap["fc"] == LayerType.LINEAR
    assert "relu" not in lmap
    assert lc.depth_of("conv") == 0.0
    assert lc.depth_of("fc") == 1.0
    try:
        lc.depth_of("relu")
        raise AssertionError("depth_of should reject unparameterized modules")
    except KeyError:
        pass

    print("\nall layer-classifier checks passed ✓")
