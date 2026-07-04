# Phase 3 gate: threshold calibration report

2026-07-04, cycle 0 (cold start). Rolling document — update each gate cycle.

## Current gate configuration (`gate_config.json`)

Relative dual-metric rule (no absolute threshold until ≥ 5 cycles, per plan):

- BLOCK if candidate GSNR < **0.8 ×** incumbent GSNR (fine-tune on QC batch,
  last block + head, 100 steps; last-layer per-sample gradient GSNR averaged
  over steps), or
- BLOCK if candidate QC accuracy < incumbent − **0.02**.
- `absolute_threshold: null` — set after ≥ 5 cycles of history in
  `results/gate_history.csv`.

GSNR is the gate metric per the plan's Phase 3 (arXiv 2001.07384: GSNR ↔
generalization gap); GradNorm is logged alongside from the same per-sample
gradients and the metric is switchable in config. The 2026-07-03 "GradNorm
not GSNR" ruling covered the unlabeled detection path (Phases 1-2), which is
untouched.

## Cycle 0 evidence (synthetic candidates vs incumbent `standin.pt`)

QC batch = 512 synthetic labeled frames (no real QC process exists yet).

| candidate | GSNR | GradNorm | QC acc | verdict | binding leg |
|---|---|---|---|---|---|
| incumbent (reference) | 0.0552 | 18.5 | 0.932 | — | — |
| healthy (continue-train, augmented) | 0.0549 | 15.6 | 0.961 | **PASS** | — |
| overfit (500 steps, 256 fixed frames) | **0.0373** | 16.5 | 0.740 | **BLOCK** | GSNR **and** accuracy |
| undertrained (15 steps) | 0.1055 | 31.2 | 0.318 | **BLOCK** | accuracy only |

## Observations

1. **GSNR separates the overfit candidate**: −32% vs incumbent, well past the
   0.8 margin — the generalization-gap signal the gate exists for, and it
   fires even though the same model's *training* loss on its memorized set is
   near zero.
2. **GSNR alone is not sufficient**: the undertrained candidate shows *higher*
   GSNR (large systematic early-training gradients) and would pass a
   GSNR-only gate. The QC-accuracy leg catches it — the plan's dual-metric
   cold-start design is necessary, not just cautious.
3. Healthy retrain sits within 1% of incumbent GSNR → 0.8 margin has
   comfortable headroom against false blocks (1 cycle of evidence only).

## Pending

- Absolute threshold: needs ≥ 5 cycles; keep relative rule until then.
- Real QC batches: gate runs on synthetic frames until a QC labeling process
  exists (blocked since 2026-07-03).
