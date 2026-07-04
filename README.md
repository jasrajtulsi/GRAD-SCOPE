# gradescope (grader-drift-pipeline)

Phases 0-3 of the conveyor-belt grading CNN failure-early-warning plan:

- **Phase 0** — `drift.py`: synthetic slow-drift corruption generator with a
  ground-truth `manifest.csv` timeline (`lighting`, `blur`, `dust`,
  `colortemp`; severity in [0,1], linear ramp, seed-reproducible).
- **Phase 1** — `detectors.py`: per-image MSP / entropy / GradNorm scores,
  rolling-window KS + CUSUM alarms against a clean deployment-start
  reference (unified `WindowDetector` interface).
- **Phase 2** — `detectors.py` + `evaluate.py`: batch-level entropy-GradNorm
  detector (Tent-loss gradient norm, **GradNorm not GSNR** per user ruling),
  same alarm framework, plus loss/T/aggregation/batch-size/param-subset
  ablations and the go/no-go gate. Full results: `results/report.md`.
- **Phase 3** — `gate.py`: supervised fragility gate for retrained
  candidates. Fine-tunes on the weekly labeled QC batch (backbone frozen),
  records last-layer per-sample gradient **GSNR** (per plan Phase 3 /
  arXiv 2001.07384; GradNorm logged alongside, metric switchable in
  `gate_config.json`). Cold-start relative dual rule: BLOCK if GSNR
  < 0.8 × incumbent or QC accuracy < incumbent − 0.02; history in
  `results/gate_history.csv`, calibration in `results/gate_calibration.md`.
- `standin.py`: no production model exists, so a small CNN trained on
  procedural "fruit on conveyor" frames (3 blemish-driven grades, clean
  accuracy 0.945) stands in.

Separate from the `cnn-grader-doctor` MCP server (analysis-only, stays thin).

## Decision log

- 2026-07-03 — Scope reduced to Phase 0 only (user). Superseded 2026-07-04:
  user ordered Phases 1-2; both delivered.
- 2026-07-03 — Gradient signal is **GradNorm, not GSNR** (user).
- 2026-07-03 — No trained production model; stand-in CNN on RTX 3090
  (PyTorch nightly cu126).
- 2026-07-03 — No real production frames/logs and no QC labeling process →
  synthetic-only validation.
- 2026-07-04 — **Go/no-go: NO-GO for the batch-level layer as primary
  detector.** Windowed GradNorm (uniform-KL, last layer) is earliest or tied
  on all 4 synthetic ramps with 0 clean false alarms; batch entropy-GradNorm
  gains 1 step on lighting but loses 7-10 steps on colortemp (the slow-drift
  target regime). Windowed GradNorm is the production detector; batch code
  stays, revisit when real degradation data exists.
- 2026-07-04 — Phase 2 config ruling: entropy-GradNorm at T=1/mean is
  unstable (9/30 clean false alarms, confidence saturation); T=2 inside the
  gradient or median aggregation fixes it at no detection-speed cost.
  Batch 256 confirmed; last-layer beats last-block.
- 2026-07-04 — **Phase 3 delivered** (user order, second scope extension).
  Gate metric is GSNR *in the supervised gate only* — the GradNorm ruling
  covered the unlabeled detection path; both statistics come from the same
  closed-form per-sample gradients and the config can switch. Cycle-0
  evidence: healthy candidate PASS; overfit candidate BLOCK on the GSNR leg
  (−32%); undertrained candidate shows *higher* GSNR and is caught by the
  accuracy leg — dual-metric rule is necessary, GSNR-only would pass it.
- 2026-07-04 — Phase 0 bug fixed: colortemp passed int pixel shifts to
  `A.RGBShift`, but albumentations reads |shift| ≤ 1 as a fraction of 255 —
  severity ~0.03 produced a full-scale +255 shift. Now continuous fractions,
  with a regression test.

## Usage

Generate a synthetic slow-drift dataset:

```
.venv\Scripts\python.exe drift.py --input <clean-frames-dir> --output out\dust ^
    --corruption dust --steps 50 --max-severity 1.0 --seed 42
```

Run the full Phase 1+2 evaluation (trains stand-in, generates 4 ramps,
scores detectors, writes `results/`):

```
.venv\Scripts\python.exe evaluate.py
```

Run the Phase 3 gate on a retrained candidate (or `--demo` to build and
gate three synthetic candidates):

```
.venv\Scripts\python.exe gate.py --candidate cand.pt --incumbent standin.pt
```

Same seed ⇒ byte-identical drift output (verified in test).

## Setup

```
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install --prerelease allow --index-url https://download.pytorch.org/whl/nightly/cu126 torch torchvision
```

## Test

```
.venv\Scripts\python.exe test_drift.py
.venv\Scripts\python.exe test_detectors.py
.venv\Scripts\python.exe test_gate.py
```

## Hard rules carried over from the plan

- **No calibration anywhere in the detection path** (no temperature scaling
  on outputs) — it destroys correct/wrong separation. Log raw logits.
  (Ablation `T` scales logits only inside gradient computation, never
  outputs.)
