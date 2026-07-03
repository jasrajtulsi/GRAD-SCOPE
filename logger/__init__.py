"""GRAD-SCOPE gradient-instrumentation package.

Exposes the gradient logger, layer classifier, and lead-time analyzer used to
capture and interpret per-layer gradient-flow dynamics during training.
"""

from .gradient_logger import GradientLogger
from .layer_classifier import (
    GradientState,
    LayerClassifier,
    LayerType,
    STATE_ORDER,
    classify,
    classify_all,
    get_state_heatmap_data,
)
from .lead_time import (
    LeadTimeAnalyzer,
    compute_all,
    detect_failure,
    detect_signal_gsnr,
    detect_signal_loss,
    detect_signal_norm,
)

__all__ = [
    "GradientLogger",
    "GradientState",
    "LayerClassifier",
    "LayerType",
    "STATE_ORDER",
    "classify",
    "classify_all",
    "get_state_heatmap_data",
    "LeadTimeAnalyzer",
    "compute_all",
    "detect_failure",
    "detect_signal_gsnr",
    "detect_signal_loss",
    "detect_signal_norm",
]
