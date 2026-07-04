"""Pure analysis logic for the GRAD-SCOPE MCP server.

This module is deliberately free of any ``mcp`` or ``torch`` dependency so it can
be unit-tested on its own and so the server stays lightweight. ``server.py`` wraps
each public function below as an MCP tool.

Data sources
------------
* ``results/live_state.json`` — a *single* snapshot the training script overwrites
  every epoch (current epoch, val accuracy, training loss, per-layer GSNR).
* ``results/*.csv`` — full per-(epoch, layer) GSNR histories written by
  ``GradientLogger.save_csv`` when an experiment completes.
* ``results/<name>_metrics.json`` (optional) — companion file for a completed
  experiment holding ``val_accs`` / ``train_losses``, used to compute true lead
  times (CSV files do not record validation accuracy).

Because ``live_state.json`` is only ever the *current* epoch, this module keeps an
in-memory history (``_HISTORY``) that accumulates as the server is polled. The MCP
stdio process is long-lived, so this history persists across tool calls for the
duration of a training run and resets automatically if a new run starts (epoch
counter goes backwards).
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

# GSNR classification thresholds — mirror logger/layer_classifier.py so the MCP
# server reports the same HEALTHY / STAGNANT / NOISY / DEAD taxonomy.
_DEAD_FLOOR = 1e-10
_STAGNANT_FLOOR = 1e-8
_DEFAULT_TAU_H = 0.1
_DEFAULT_TAU_N = 0.01

# In-memory accumulation of per-epoch snapshots for the *live* run.
# Each entry: {"epoch": int, "val_accuracy": float|None,
#              "training_loss": float|None, "current_gsnr": {layer: float}}
_HISTORY: list[dict] = []


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #

def results_dir() -> Path:
    """Directory holding live_state.json and experiment CSVs.

    Defaults to ``<repo-root>/results`` (the repo root is the parent of the
    ``gradscope-mcp`` folder). Override with ``GRADSCOPE_RESULTS_DIR``.
    """
    env = os.environ.get("GRADSCOPE_RESULTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "results"


def live_state_path() -> Path:
    return results_dir() / "live_state.json"


# --------------------------------------------------------------------------- #
# Loading the live snapshot                                                   #
# --------------------------------------------------------------------------- #

def read_live_state() -> dict:
    """Read and parse live_state.json. Raises if missing or unparseable."""
    with open(live_state_path()) as f:
        return json.load(f)


def _load() -> tuple[Optional[dict], Optional[dict]]:
    """Return ``(state, error)`` — exactly one is non-None."""
    try:
        return read_live_state(), None
    except FileNotFoundError:
        return None, {
            "error": (
                f"live_state.json not found at {live_state_path()}. "
                "Start a GRAD-SCOPE training run — it writes this file every epoch."
            )
        }
    except json.JSONDecodeError:
        return None, {
            "error": (
                "live_state.json exists but could not be parsed "
                "(it may be mid-write); try again in a moment."
            )
        }


def _record_history(state: dict) -> None:
    """Append the current snapshot to the in-memory history (deduped by epoch)."""
    epoch = state.get("epoch")
    if epoch is None:
        return
    if _HISTORY:
        last_epoch = _HISTORY[-1]["epoch"]
        if epoch == last_epoch:
            # Same epoch already recorded — refresh it in place.
            _HISTORY[-1] = _snapshot(state)
            return
        if epoch < last_epoch:
            # Epoch counter went backwards: a new training run started. Reset.
            _HISTORY.clear()
    _HISTORY.append(_snapshot(state))


def _snapshot(state: dict) -> dict:
    gsnr = state.get("current_gsnr") or {}
    return {
        "epoch": state.get("epoch"),
        "val_accuracy": state.get("val_accuracy"),
        "training_loss": state.get("training_loss"),
        "current_gsnr": {k: float(v) for k, v in gsnr.items() if v is not None},
    }


def reset_history() -> None:
    """Clear the in-memory history (used by tests)."""
    _HISTORY.clear()


# --------------------------------------------------------------------------- #
# Classification & detection (local copies of the logger's logic)             #
# --------------------------------------------------------------------------- #

def classify_gsnr(
    values: list[float],
    patience: int = 5,
    tau_h: float = _DEFAULT_TAU_H,
    tau_n: float = _DEFAULT_TAU_N,
) -> str:
    """Classify one layer's GSNR history as HEALTHY/STAGNANT/NOISY/DEAD."""
    if not values:
        return "DEAD"
    current = values[-1]
    window = values[-patience:]
    if len(window) >= patience and all(v < _DEAD_FLOOR for v in window):
        return "DEAD"
    if current < _STAGNANT_FLOOR:
        return "STAGNANT"
    if current < tau_n:
        return "NOISY"
    if current >= tau_h:
        return "HEALTHY"
    return "NOISY"


def _detect_failure(
    val_accs: list[float], drop_threshold: float = 0.05, patience: int = 3
) -> Optional[int]:
    """First index of a sustained (>drop_threshold relative) drop from peak."""
    peak = float("-inf")
    run_start: Optional[int] = None
    run_len = 0
    for i, acc in enumerate(val_accs):
        peak = max(peak, acc)
        if acc < peak * (1.0 - drop_threshold):
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= patience:
                return run_start
        else:
            run_len = 0
            run_start = None
    return None


def _states_timeline(history: list[dict]) -> list[tuple[int, dict]]:
    """Per-epoch ``(epoch, {layer: state})`` using each layer's history so far."""
    per_layer: dict[str, list[float]] = {}
    timeline: list[tuple[int, dict]] = []
    for snap in history:
        for layer, val in snap["current_gsnr"].items():
            per_layer.setdefault(layer, []).append(float(val))
        states = {layer: classify_gsnr(vals) for layer, vals in per_layer.items()}
        timeline.append((snap["epoch"], states))
    return timeline


def _gsnr_signal_epoch(timeline: list[tuple[int, dict]]) -> Optional[int]:
    """First epoch at which any layer is NOISY or DEAD."""
    for epoch, states in timeline:
        if any(s in ("NOISY", "DEAD") for s in states.values()):
            return epoch
    return None


def _failure_epoch(history: list[dict]) -> Optional[int]:
    """Epoch of the validation-accuracy collapse, or None."""
    pairs = [
        (h["epoch"], h["val_accuracy"])
        for h in history
        if h.get("val_accuracy") is not None
    ]
    if not pairs:
        return None
    idx = _detect_failure([a for _, a in pairs])
    return pairs[idx][0] if idx is not None else None


# --------------------------------------------------------------------------- #
# Tool implementations                                                        #
# --------------------------------------------------------------------------- #

def get_current_gsnr() -> dict:
    """Layer -> current GSNR value from live_state.json."""
    state, error = _load()
    if error:
        return error
    _record_history(state)
    return {k: float(v) for k, v in (state.get("current_gsnr") or {}).items()}


def get_layer_states() -> dict:
    """Layer -> current state (HEALTHY / STAGNANT / NOISY / DEAD)."""
    state, error = _load()
    if error:
        return error
    _record_history(state)
    timeline = _states_timeline(_HISTORY)
    return dict(timeline[-1][1]) if timeline else {}


def get_training_status() -> dict:
    """Current epoch, totals, metrics, and whether failure/GSNR signals fired."""
    state, error = _load()
    if error:
        return error
    _record_history(state)
    timeline = _states_timeline(_HISTORY)
    signal_epoch = _gsnr_signal_epoch(timeline)
    failure_epoch = _failure_epoch(_HISTORY)
    total_epochs = state.get("total_epochs")
    if total_epochs is None and os.environ.get("GRADSCOPE_TOTAL_EPOCHS"):
        total_epochs = int(os.environ["GRADSCOPE_TOTAL_EPOCHS"])
    return {
        "current_epoch": state.get("epoch"),
        "total_epochs": total_epochs,
        "val_accuracy": state.get("val_accuracy"),
        "training_loss": state.get("training_loss"),
        "failure_detected": failure_epoch is not None,
        "gsnr_signal_fired": signal_epoch is not None,
    }


def get_lead_time_estimate() -> dict:
    """Lead time of the GSNR early-warning signal over the observed failure."""
    state, error = _load()
    if error:
        return error
    _record_history(state)
    timeline = _states_timeline(_HISTORY)
    signal_epoch = _gsnr_signal_epoch(timeline)
    failure_epoch = _failure_epoch(_HISTORY)

    if signal_epoch is not None and failure_epoch is not None:
        lead = failure_epoch - signal_epoch
        return {
            "status": "detected",
            "lead_time": lead,
            "signal_epoch": signal_epoch,
            "failure_epoch": failure_epoch,
            "message": (
                f"GSNR signal fired at epoch {signal_epoch}; failure at epoch "
                f"{failure_epoch}; lead time = {lead} epochs."
            ),
        }
    if signal_epoch is not None:
        return {
            "status": "waiting",
            "lead_time": None,
            "signal_epoch": signal_epoch,
            "failure_epoch": None,
            "message": f"Signal fired at epoch {signal_epoch}, waiting for failure.",
        }
    return {
        "status": "not_detected",
        "lead_time": None,
        "signal_epoch": None,
        "failure_epoch": None,
        "message": "Not detected yet.",
    }


def _summarize_csv(csv_path: Path) -> dict:
    """Summarize one completed experiment CSV (+ optional metrics companion)."""
    by_epoch: dict[int, dict[str, float]] = {}
    layers: set[str] = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                epoch = int(row["epoch"])
                layer = row["layer"]
                gsnr = float(row["gsnr"])
            except (KeyError, ValueError):
                continue
            by_epoch.setdefault(epoch, {})[layer] = gsnr
            layers.add(layer)

    history = [{"epoch": e, "current_gsnr": by_epoch[e]} for e in sorted(by_epoch)]
    timeline = _states_timeline(history)
    final_states = dict(timeline[-1][1]) if timeline else {}
    signal_epoch = _gsnr_signal_epoch(timeline)

    summary: dict[str, Any] = {
        "experiment": csv_path.stem,
        "num_epochs": len(by_epoch),
        "layers": sorted(layers),
        "final_states": final_states,
        "gsnr_signal_epoch": signal_epoch,
        "failure_epoch": None,
        "lead_time": None,
        "note": None,
    }

    # Companion metrics file supplies validation accuracy for true lead time.
    metrics_path = csv_path.with_name(f"{csv_path.stem}_metrics.json")
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {}
        val_accs = metrics.get("val_accs") or metrics.get("val_accuracies")
        if val_accs:
            failure_epoch = _detect_failure([float(a) for a in val_accs])
            summary["failure_epoch"] = failure_epoch
            if failure_epoch is not None and signal_epoch is not None:
                summary["lead_time"] = failure_epoch - signal_epoch
                summary["note"] = "lead_time = failure_epoch - gsnr_signal_epoch"
            elif failure_epoch is None:
                summary["note"] = "no accuracy collapse detected in this run"
    else:
        summary["note"] = (
            f"no {metrics_path.name} companion found; reported GSNR signal epoch "
            "only (CSV files do not record validation accuracy)"
        )
    return summary


def get_all_results() -> dict:
    """Summarize every completed experiment CSV in the results folder."""
    rdir = results_dir()
    if not rdir.exists():
        return {
            "error": f"results directory not found at {rdir}.",
            "num_experiments": 0,
            "experiments": {},
        }
    experiments: dict[str, dict] = {}
    for csv_path in sorted(rdir.glob("*.csv")):
        try:
            experiments[csv_path.stem] = _summarize_csv(csv_path)
        except Exception as exc:  # keep one bad file from sinking the whole call
            experiments[csv_path.stem] = {
                "experiment": csv_path.stem,
                "error": f"failed to parse: {exc}",
            }
    return {
        "results_dir": str(rdir),
        "num_experiments": len(experiments),
        "experiments": experiments,
    }
