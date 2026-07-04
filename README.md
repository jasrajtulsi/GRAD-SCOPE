# grader-drift-pipeline

Phase 0 of the conveyor-belt grading CNN failure-early-warning plan: synthetic
slow-drift replay data with a ground-truth corruption timeline.

Separate from the `cnn-grader-doctor` MCP server (analysis-only, stays thin).
Later phases (windowed MSP/entropy baseline, batch-level gradient detector)
land here too.

## Decision log

- 2026-07-03 — Scope reduced to Phase 0 only (user).
- 2026-07-03 — Gradient signal will be **GradNorm, not GSNR** (user).
- 2026-07-03 — No trained production model exists; PyTorch nightly cu126
  installed for a stand-in model in later phases (RTX 3090).
- 2026-07-03 — No real production frames/logs and **no QC labeling process
  exists** → the "real degradation period" ground-truth curve from the plan is
  blocked; Phase 0 delivers the synthetic side only.

## Usage

```
.venv\Scripts\python.exe drift.py --input <clean-frames-dir> --output out\dust ^
    --corruption dust --steps 50 --max-severity 1.0 --seed 42
```

Corruptions: `lighting`, `blur`, `dust`, `colortemp` — severity ramps linearly
from 0 (step 0, clean baseline) to `--max-severity`. Output: one dir per step
plus `manifest.csv` (step, severity, corruption, source_frame, output_frame,
seed) — the ground-truth timeline detectors are scored against.

Same seed ⇒ byte-identical output (verified in test).

## Setup

```
uv venv --python 3.12
uv pip install -r requirements.txt
# torch (later phases, stand-in model):
uv pip install --prerelease allow --index-url https://download.pytorch.org/whl/nightly/cu126 torch torchvision
```

## Test

```
.venv\Scripts\python.exe test_drift.py
```

## Hard rules carried over from the plan

- **No calibration anywhere in the detection path** (no temperature scaling on
  outputs) — it destroys correct/wrong separation. Log raw logits.
