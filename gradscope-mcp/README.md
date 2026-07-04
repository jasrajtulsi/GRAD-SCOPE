# gradscope-mcp

An [MCP](https://modelcontextprotocol.io) server for **live monitoring of GRAD-SCOPE
training runs**. It surfaces per-layer gradient signal-to-noise ratios (GSNR),
layer health states, training status, and early-warning lead times to any MCP
client (Claude Desktop, Claude Code, etc.) over **stdio**.

> **This server is specific to GRAD-SCOPE and is separate from
> `cnn-grader-doctor`.** `cnn-grader-doctor` diagnoses trained CNN *grader* models
> (confusion matrices, grade distributions, image quality). `gradscope-mcp`
> instead watches a *training run in progress* through its gradient-flow
> instrumentation. The two servers share no tools, data, or state, and are
> registered independently in your MCP client.

## What it reads

The GRAD-SCOPE training script (via `logger.GradientLogger`) writes
`results/live_state.json` **every epoch** — a single snapshot of the current epoch,
validation accuracy, training loss, and per-layer GSNR. This server reads that file
plus the experiment CSV files in `results/`.

Because `live_state.json` is only ever the *current* epoch (it is overwritten each
epoch), the server keeps an **in-memory history** that accumulates as it is polled.
The stdio server process is long-lived, so this history persists for the duration of
a training run and resets automatically when a new run starts (the epoch counter
goes backwards). Failure detection and lead-time estimates are derived from this
accumulated history, so poll the server periodically (e.g. once per epoch) for the
richest results.

## Tools

| Tool | Returns |
| --- | --- |
| `get_current_gsnr()` | `{layer_name: current GSNR}`. Error message if `live_state.json` is missing. |
| `get_layer_states()` | `{layer_name: state}` where state ∈ `HEALTHY`, `STAGNANT`, `NOISY`, `DEAD`. |
| `get_training_status()` | `current_epoch`, `total_epochs`, `val_accuracy`, `training_loss`, `failure_detected` (bool), `gsnr_signal_fired` (bool). |
| `get_lead_time_estimate()` | Lead-time number if both signal and failure seen; a "signal fired at epoch X, waiting for failure" message if only the signal fired; "not detected yet" otherwise. |
| `get_all_results()` | Summary of every completed experiment CSV in `results/`, with per-experiment lead times. |

### State classification

Layer states use the same thresholds as `logger/layer_classifier.py`:

- **DEAD** — GSNR below `1e-10` for the last 5 epochs (sustained flatline).
- **STAGNANT** — current GSNR below `1e-8`.
- **NOISY** — current GSNR below `tau_h = 0.1` but not stagnant/dead.
- **HEALTHY** — current GSNR at or above `tau_h = 0.1`.

The **GSNR signal** is considered fired once any layer is `NOISY` or `DEAD`. A
**failure** is a sustained (>5% relative) drop in validation accuracy from its peak
over 3 consecutive epochs. **Lead time = failure epoch − signal epoch** (positive =
the gradient signal warned that many epochs in advance).

## Configuration

| Env var | Purpose | Default |
| --- | --- | --- |
| `GRADSCOPE_RESULTS_DIR` | Directory containing `live_state.json` and CSVs | `<repo-root>/results` |
| `GRADSCOPE_TOTAL_EPOCHS` | Fallback for `total_epochs` if not in `live_state.json` | unset |

### `get_all_results` and lead time

Experiment CSVs (`GradientLogger.save_csv`) record per-`(epoch, layer)` GSNR but
**not** validation accuracy, so lead time cannot be computed from a CSV alone. To
get a true lead time for a completed experiment, save a companion
`results/<name>_metrics.json` alongside `results/<name>.csv`:

```json
{ "val_accs": [0.40, 0.55, 0.68, 0.75, 0.80, 0.70, 0.64, 0.58], "train_losses": [2.0, 1.5, 1.1, 0.9, 0.85, 0.95, 1.2, 1.5] }
```

Without it, `get_all_results` still reports each experiment's GSNR signal epoch and
final layer states, with `lead_time: null`.

## Running

Install dependencies and run:

```bash
cd gradscope-mcp
uv run server.py          # or: pip install "mcp[cli]>=1.4.0" && python server.py
```

### Register with an MCP client

Claude Desktop (`claude_desktop_config.json`) or Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "gradscope": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/GRAD-SCOPE/gradscope-mcp", "server.py"],
      "env": { "GRADSCOPE_RESULTS_DIR": "/absolute/path/to/GRAD-SCOPE/results" }
    }
  }
}
```

## Layout

```
gradscope-mcp/
├── server.py        # FastMCP server: the 5 tools + stdio transport
├── analysis.py      # pure logic (no mcp/torch deps) — unit-testable
├── pyproject.toml   # declares the `mcp` dependency
└── README.md
```

`analysis.py` is intentionally dependency-free (no `mcp`, no `torch`) so the
detection logic can be tested in isolation; `server.py` only adds the MCP tool
wrappers and transport.
