"""GRAD-SCOPE MCP server.

A FastMCP server (stdio transport) that exposes the live state of a GRAD-SCOPE
training run to an MCP client. It reads ``results/live_state.json`` — the snapshot
the training script rewrites every epoch — plus the experiment CSVs, and reports
per-layer gradient signal-to-noise ratios (GSNR), layer health states, training
status, and early-warning lead times.

This server is for GRAD-SCOPE gradient-flow monitoring only. It is entirely
separate from the ``cnn-grader-doctor`` server (which diagnoses CNN grader models);
they share no tools, data, or state.

Run directly:   python server.py
Or with uv:     uv run server.py
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import analysis

mcp = FastMCP("gradscope-mcp")


@mcp.tool()
def get_current_gsnr() -> dict:
    """Return the current gradient signal-to-noise ratio (GSNR) per layer.

    Reads the latest ``results/live_state.json`` snapshot and returns a mapping
    of ``layer_name -> current GSNR value``. Returns an ``{"error": ...}`` message
    if the file has not been written yet (no training run is active).
    """
    return analysis.get_current_gsnr()


@mcp.tool()
def get_layer_states() -> dict:
    """Return each layer's current gradient-health state.

    Maps ``layer_name -> one of HEALTHY, STAGNANT, NOISY, DEAD``, classified from
    the layer's GSNR history observed so far this run.
    """
    return analysis.get_layer_states()


@mcp.tool()
def get_training_status() -> dict:
    """Return an overview of the running training job.

    Includes ``current_epoch``, ``total_epochs``, ``val_accuracy``,
    ``training_loss``, and the booleans ``failure_detected`` (validation accuracy
    has collapsed) and ``gsnr_signal_fired`` (a layer has gone NOISY or DEAD).
    """
    return analysis.get_training_status()


@mcp.tool()
def get_lead_time_estimate() -> dict:
    """Estimate how early the GSNR signal warned of failure.

    * If both the GSNR signal and a failure have been observed, returns the lead
      time (epochs the signal preceded the failure).
    * If only the signal has fired, reports that it fired at epoch X and is
      waiting for failure.
    * If neither has occurred, reports that nothing is detected yet.
    """
    return analysis.get_lead_time_estimate()


@mcp.tool()
def get_all_results() -> dict:
    """Summarize all completed experiments from the results CSV files.

    Returns per-experiment summaries (epochs, layers, final states, GSNR signal
    epoch, and lead time where a ``<name>_metrics.json`` companion supplies
    validation accuracy).
    """
    return analysis.get_all_results()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
