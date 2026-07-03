"""GRAD-SCOPE gradient-instrumentation package.

Exposes the gradient logger, layer classifier, and lead-time analyzer used to
capture and interpret per-layer gradient-flow dynamics during training.
"""

from .gradient_logger import GradientLogger
from .layer_classifier import LayerClassifier, LayerType
from .lead_time import LeadTimeAnalyzer

__all__ = [
    "GradientLogger",
    "LayerClassifier",
    "LayerType",
    "LeadTimeAnalyzer",
]
