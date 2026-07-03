"""Lead-time analysis for GRAD-SCOPE.

Quantifies how far in advance gradient-flow anomalies (vanishing/exploding
signatures, collapse) precede an observable drop in validation performance —
the "lead time" of the early-warning signal.

Stub only — no implementation yet.
"""

from __future__ import annotations

from typing import Any


class LeadTimeAnalyzer:
    """Measure lead time between gradient-flow anomalies and performance drops."""

    def __init__(self, gradient_records: Any, performance_records: Any) -> None:
        raise NotImplementedError

    def detect_anomalies(self) -> Any:
        """Return the steps at which gradient-flow anomalies are detected."""
        raise NotImplementedError

    def detect_performance_drops(self) -> Any:
        """Return the steps at which validation performance degrades."""
        raise NotImplementedError

    def compute_lead_time(self) -> Any:
        """Return the lead time (in steps/epochs) for each detected event."""
        raise NotImplementedError
