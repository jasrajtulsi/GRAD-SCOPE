"""Command-line entry point for GRAD-SCOPE experiments.

Parses arguments, builds a BaselineConfig, and dispatches to the selected
experiment runner.

Stub only — no implementation yet.
"""

from __future__ import annotations

import argparse

from experiments.baseline import BaselineConfig, run_baseline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments into a namespace."""
    raise NotImplementedError


def main() -> None:
    """Build the config from CLI args and launch the experiment."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
