"""Lead-time analysis for GRAD-SCOPE.

Quantifies how far in advance a gradient-flow early-warning signal precedes an
observable drop in validation performance — the "lead time" of the signal.

The module offers three independent early-warning detectors and one failure
detector, all operating on per-epoch sequences:

* :func:`detect_failure` — when validation accuracy actually collapses.
* :func:`detect_signal_gsnr` — when a layer's GSNR state first turns NOISY/DEAD.
* :func:`detect_signal_norm` — when a layer's gradient norm first vanishes.
* :func:`detect_signal_loss` — when training loss first stops improving.

:func:`compute_all` ties them together and reports, for each signal, how many
epochs earlier than the failure it fired (a positive lead time = advance warning).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence


# --------------------------------------------------------------------------- #
# Failure detection                                                           #
# --------------------------------------------------------------------------- #

def detect_failure(
    val_accs: Sequence[float],
    drop_threshold: float = 0.05,
    patience: int = 3,
) -> Optional[int]:
    """Return the first epoch at which validation accuracy has collapsed.

    A collapse is a *relative* drop of more than ``drop_threshold`` (default 5%)
    below the best accuracy seen so far, sustained for ``patience`` consecutive
    epochs. The returned epoch is the first epoch of that qualifying run, i.e.
    the onset of the sustained drop.

    Returns ``None`` if accuracy never sustains such a drop.
    """
    peak = float("-inf")
    run_start: Optional[int] = None
    run_len = 0

    for i, acc in enumerate(val_accs):
        peak = max(peak, acc)  # running peak includes the current epoch
        dropped = acc < peak * (1.0 - drop_threshold)
        if dropped:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= patience:
                return run_start
        else:
            run_len = 0
            run_start = None
    return None


# --------------------------------------------------------------------------- #
# Early-warning signal detectors                                              #
# --------------------------------------------------------------------------- #

def _iter_states(per_epoch_entry: Any) -> Iterable[str]:
    """Yield state strings from one epoch's layer-state collection.

    Accepts a ``{layer_name: state}`` mapping or any iterable of states, where a
    state is a ``GradientState`` (a str subclass), its ``.value``, or a plain
    string.
    """
    if hasattr(per_epoch_entry, "values"):
        entries: Iterable[Any] = per_epoch_entry.values()
    elif isinstance(per_epoch_entry, str):
        entries = [per_epoch_entry]
    else:
        entries = per_epoch_entry
    for state in entries:
        yield str(getattr(state, "value", state)).upper()


def detect_signal_gsnr(layer_states_per_epoch: Sequence[Any]) -> Optional[int]:
    """Return the first epoch at which *any* layer is NOISY or DEAD.

    ``layer_states_per_epoch`` is one entry per epoch; each entry is a mapping
    of layer name to :class:`~logger.layer_classifier.GradientState` (or an
    iterable of such states).

    Returns ``None`` if no layer is ever NOISY or DEAD.
    """
    for epoch, entry in enumerate(layer_states_per_epoch):
        if any(state in ("NOISY", "DEAD") for state in _iter_states(entry)):
            return epoch
    return None


def detect_signal_norm(
    layer_norms_per_epoch: Sequence[Any],
    threshold: float = 1e-4,
) -> Optional[int]:
    """Return the first epoch at which *any* layer's gradient norm vanishes.

    ``layer_norms_per_epoch`` is one entry per epoch; each entry is a mapping of
    layer name to gradient norm (or an iterable of norms). A norm at or below
    ``threshold`` counts as vanished.

    Returns ``None`` if no layer's norm ever falls to the threshold.
    """
    for epoch, entry in enumerate(layer_norms_per_epoch):
        norms = entry.values() if hasattr(entry, "values") else entry
        if any(float(n) <= threshold for n in norms):
            return epoch
    return None


def detect_signal_loss(train_losses: Sequence[float]) -> Optional[int]:
    """Return the first epoch at which training loss stops improving.

    The signal fires at the first epoch whose loss is non-finite (NaN/inf), or
    whose loss rises back above the best (minimum) loss achieved in any earlier
    epoch — i.e. the onset of divergence or a plateau reversal.

    Returns ``None`` if the loss only ever improves (or stays flat below its best).
    """
    best = float("inf")
    for epoch, loss in enumerate(train_losses):
        loss = float(loss)
        if not math.isfinite(loss):
            return epoch
        if epoch > 0 and loss > best:
            return epoch
        best = min(best, loss)
    return None


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #

def _lead_time(failure: Optional[int], signal: Optional[int]) -> Optional[int]:
    """Epochs by which ``signal`` preceded ``failure`` (positive = advance warning)."""
    if failure is None or signal is None:
        return None
    return failure - signal


def compute_all(
    val_accs: Sequence[float],
    layer_states: Sequence[Any],
    layer_norms: Sequence[Any],
    train_losses: Sequence[float],
) -> dict:
    """Run every detector and report each signal's lead time over the failure.

    Returns a dict with:

    * ``failure_epoch`` — epoch of the validation-accuracy collapse (or ``None``).
    * ``gsnr_signal`` / ``norm_signal`` / ``loss_signal`` — epoch each early-warning
      signal first fired (or ``None``).
    * ``gsnr_lead_time`` / ``norm_lead_time`` / ``loss_lead_time`` — ``failure_epoch``
      minus the corresponding signal epoch. Positive means the signal preceded
      the failure by that many epochs; ``None`` if either the failure or the
      signal never occurred.
    """
    failure_epoch = detect_failure(val_accs)
    gsnr_signal = detect_signal_gsnr(layer_states)
    norm_signal = detect_signal_norm(layer_norms)
    loss_signal = detect_signal_loss(train_losses)

    return {
        "failure_epoch": failure_epoch,
        "gsnr_signal": gsnr_signal,
        "norm_signal": norm_signal,
        "loss_signal": loss_signal,
        "gsnr_lead_time": _lead_time(failure_epoch, gsnr_signal),
        "norm_lead_time": _lead_time(failure_epoch, norm_signal),
        "loss_lead_time": _lead_time(failure_epoch, loss_signal),
    }


class LeadTimeAnalyzer:
    """Measure lead time between gradient-flow anomalies and performance drops.

    ``gradient_records`` — one entry per epoch, each a ``{layer: state}`` mapping
    (or iterable of states) in the same form :func:`detect_signal_gsnr` accepts.
    ``performance_records`` — per-epoch validation accuracies.
    """

    def __init__(self, gradient_records: Any, performance_records: Any) -> None:
        self.gradient_records = list(gradient_records)
        self.performance_records = [float(a) for a in performance_records]

    def detect_anomalies(self) -> list[int]:
        """Return every epoch at which any layer's gradient is NOISY or DEAD."""
        return [
            epoch
            for epoch, entry in enumerate(self.gradient_records)
            if any(state in ("NOISY", "DEAD") for state in _iter_states(entry))
        ]

    def detect_performance_drops(self) -> list[int]:
        """Return the onset epoch of the validation collapse ([] if none)."""
        failure = detect_failure(self.performance_records)
        return [] if failure is None else [failure]

    def compute_lead_time(self) -> Optional[int]:
        """Epochs by which the first anomaly preceded the performance collapse.

        Positive means advance warning; ``None`` if either never occurred.
        """
        anomalies = self.detect_anomalies()
        drops = self.detect_performance_drops()
        if not anomalies or not drops:
            return None
        return drops[0] - anomalies[0]


if __name__ == "__main__":
    # A run that climbs, then collapses at epoch 6. The gradient signals fire a
    # few epochs earlier than the accuracy actually falls apart.
    val_accs = [0.40, 0.55, 0.68, 0.75, 0.80, 0.81, 0.70, 0.64, 0.58, 0.55]
    #            0     1     2     3     4     5     6     7     8     9
    # Peak 0.81 at epoch 5; a >5% relative drop (< 0.7695) starts at epoch 6 and
    # holds for epochs 6,7,8 -> failure onset at epoch 6.

    # Per-epoch layer states: a layer turns NOISY at epoch 3, well before failure.
    layer_states = [
        {"l0": "HEALTHY", "l1": "HEALTHY"},   # 0
        {"l0": "HEALTHY", "l1": "HEALTHY"},   # 1
        {"l0": "HEALTHY", "l1": "HEALTHY"},   # 2
        {"l0": "HEALTHY", "l1": "NOISY"},     # 3  <- first NOISY
        {"l0": "NOISY",   "l1": "NOISY"},     # 4
        {"l0": "NOISY",   "l1": "DEAD"},      # 5
        {"l0": "DEAD",    "l1": "DEAD"},      # 6
        {"l0": "DEAD",    "l1": "DEAD"},      # 7
        {"l0": "DEAD",    "l1": "DEAD"},      # 8
        {"l0": "DEAD",    "l1": "DEAD"},      # 9
    ]

    # Per-epoch gradient norms: l1's norm vanishes (<= 1e-4) at epoch 4.
    layer_norms = [
        {"l0": 1.0e-1, "l1": 1.0e-1},   # 0
        {"l0": 5.0e-2, "l1": 5.0e-2},   # 1
        {"l0": 2.0e-2, "l1": 1.0e-2},   # 2
        {"l0": 1.0e-2, "l1": 1.0e-3},   # 3
        {"l0": 5.0e-3, "l1": 9.0e-5},   # 4  <- first vanished norm
        {"l0": 2.0e-3, "l1": 1.0e-5},   # 5
        {"l0": 1.0e-3, "l1": 1.0e-6},   # 6
        {"l0": 5.0e-4, "l1": 1.0e-7},   # 7
        {"l0": 1.0e-4, "l1": 1.0e-8},   # 8
        {"l0": 1.0e-5, "l1": 1.0e-9},   # 9
    ]

    # Training loss decreases, then rises back above its best at epoch 5.
    train_losses = [2.0, 1.5, 1.1, 0.9, 0.85, 0.95, 1.2, 1.5, 1.8, 2.1]
    #                0    1    2    3    4     5     6    7    8    9

    result = compute_all(val_accs, layer_states, layer_norms, train_losses)

    print("individual detectors:")
    print("  detect_failure     :", detect_failure(val_accs))
    print("  detect_signal_gsnr :", detect_signal_gsnr(layer_states))
    print("  detect_signal_norm :", detect_signal_norm(layer_norms))
    print("  detect_signal_loss :", detect_signal_loss(train_losses))

    print("\ncompute_all():")
    for key, value in result.items():
        print(f"  {key:16}: {value}")

    assert result["failure_epoch"] == 6
    assert result["gsnr_signal"] == 3
    assert result["norm_signal"] == 4
    assert result["loss_signal"] == 5
    assert result["gsnr_lead_time"] == 3   # NOISY 3 epochs before the collapse
    assert result["norm_lead_time"] == 2
    assert result["loss_lead_time"] == 1

    # Edge cases: a stable run has no failure and no signals.
    stable = [0.4, 0.5, 0.6, 0.7, 0.75, 0.78]
    assert detect_failure(stable) is None
    assert detect_signal_gsnr([{"l0": "HEALTHY"}] * 6) is None
    assert detect_signal_norm([{"l0": 1.0}] * 6) is None
    assert detect_signal_loss([2.0, 1.5, 1.0, 0.8, 0.7, 0.6]) is None
    none_result = compute_all(
        stable, [{"l0": "HEALTHY"}] * 6, [{"l0": 1.0}] * 6, [2.0, 1.5, 1.0, 0.8, 0.7, 0.6]
    )
    assert all(none_result[k] is None for k in none_result), none_result
    # NaN loss is flagged immediately.
    assert detect_signal_loss([1.0, float("nan"), 0.5]) == 1

    # LeadTimeAnalyzer wraps the same detectors.
    analyzer = LeadTimeAnalyzer(layer_states, val_accs)
    assert analyzer.detect_anomalies()[0] == 3
    assert analyzer.detect_performance_drops() == [6]
    assert analyzer.compute_lead_time() == 3
    stable_analyzer = LeadTimeAnalyzer([{"l0": "HEALTHY"}] * 6, stable)
    assert stable_analyzer.detect_anomalies() == []
    assert stable_analyzer.detect_performance_drops() == []
    assert stable_analyzer.compute_lead_time() is None
    print("LeadTimeAnalyzer: lead time =", analyzer.compute_lead_time())

    print("\nall lead-time checks passed ✓")
