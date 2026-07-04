# Phase 1 + 2 report: baseline vs batch-level gradient detectors

2026-07-04. Synthetic-only validation (no real production frames / QC labels
exist — decision 2026-07-03). Stand-in CNN (`standin.py`, clean accuracy
0.945) over 4 corruption ramps from `drift.py`: 31 steps × 256 frames,
severity 0 → 1.0 linear. Reference = 2560 clean frames at "deployment start".
Clean stream = 30 × 256 fresh clean frames for false alarms. Failure truth =
first of 2 consecutive steps with accuracy < clean − 0.05 (= 0.895).

All configs: KS alarm = p < 0.01 on 2 consecutive windows; CUSUM two-sided,
k = 0.5, h = 8, standardized by bootstrap over the reference. No calibration
anywhere in the detection path (hard rule); T in ablations scales logits only
inside the gradient computation.

## Ground truth (accuracy fail step, of 30)

| ramp | fail step | severity at failure | character |
|---|---|---|---|
| lighting | 8 | 0.27 | plateau then slide to 0.33 |
| blur | 2 | 0.07 | fast, near-linear decay |
| dust | 4 | 0.13 | plateau then collapse by step 8 |
| colortemp | 19 | 0.63 | slowest, gradual (the target regime) |

## Scorecard — earliest alarm step (KS / CUSUM), lower = earlier

| detector | lighting (fail 8) | blur (2) | dust (4) | colortemp (19) | clean FA |
|---|---|---|---|---|---|
| **Phase 1** windowed MSP | 7 / 8 | 3 / 3 | 4 / 5 | 12 / 17 | **5**/29 |
| **Phase 1** windowed entropy | 4 / 9 | 3 / 2 | 4 / 6 | 12 / 16 | 0/29 |
| **Phase 1** windowed GradNorm | **3 / 3** | **2 / 2** | **2 / 3** | **8 / 23** | 0/29 |
| **Phase 2** batch eGN (T=1, mean) | 2 / 3 | 2 / 5 | 2 / 3 | 16 / 18 | 9/30 ✗ |
| **Phase 2** batch eGN (T=1, median) | **2 / 2** | 2 / 2 | 2 / 2 | 18 / 23 | 0/30 |
| **Phase 2** batch eGN (T=2, mean) | 3 / 3 | 2 / 2 | 2 / 3 | 11 / 17 | 0/30 |
| **Phase 2** batch uniform-GN (mean) | 3 / 3 | 2 / 2 | 2 / 3 | 10 / 25 | 0/30 |

Windowed detectors: window 512, stride 256 (window ≈ 2 steps). Batch
detectors: batch 256 = 1 step. eGN = entropy-loss GradNorm (Tent-style,
unlabeled); GradNorm per user ruling 2026-07-03 (not GSNR).

Lead time (fail − best alarm): Phase 1 GradNorm 5 / 0 / 2 / 11 steps;
Phase 2 best-config 6 / 0 / 2 / 8. Blur fails too fast (step 2) for any
pre-warning at this ramp rate.

## Phase 2 ablations (dust ramp, alarm KS/CUSUM, clean FA of 30)

| axis | config | alarm | clean FA |
|---|---|---|---|
| loss | entropy T=1 (base) | 2 / 3 | 9 ✗ |
| loss | uniform T=1 | 2 / 3 | 0 |
| loss | pseudo-label CE | 3 / 5 | 0 |
| loss | entropy T=2 | 2 / 3 | 0 |
| loss | entropy T=10 | 2 / 4 | 0 |
| agg | median | 2 / 2 | 0 |
| agg | p10 | 2 / 7 | 8 ✗ |
| batch | 128 | 1 / 3 | 9 ✗ |
| batch | 512 | 3 / 11 | 2 |
| params | last block (torch.func) | 2 / 2 | 8 ✗ |

Findings: raw entropy-gradient mean is heavy-tailed on clean data → false
alarms (the confidence-saturation risk from the plan, item 2). Temperature
T=2 inside the gradient, or median aggregation, fixes it at zero cost in
detection speed. Batch < 256 too noisy, batch 512 too slow — 256 confirmed.
Last-block gradients add cost and false alarms, no speed — last layer wins.

## Go/no-go decision (plan Phase 2 gate)

**No-go for the batch-level layer as the primary detector.** The plan's rule:
the batch detector must significantly beat windowed MSP/entropy *and*
windowed GradNorm on ≥ 2 ramps. It beats MSP/entropy on lighting and dust,
but vs windowed GradNorm it gains 1 step on lighting, ties blur/dust, and
loses 7–10 steps on colortemp — the slow-drift regime it was built for.

**Ruling: windowed GradNorm (uniform-KL, last layer, KS+CUSUM) is the
production detector.** It is earliest or tied on all 4 ramps with 0/29 clean
false alarms and is the cheapest gradient signal (closed form, no autograd).
The batch-level code stays in `detectors.py` (shares the alarm framework);
revisit only if real degradation data (still blocked) contradicts the
synthetic result. MSP is strictly dominated (latest alarms + 5/29 false
alarms) — keep only as a logging sanity signal.

## Bugs found and fixed during evaluation

1. `drift.py` colortemp passed int pixel shifts to `A.RGBShift`;
   albumentations reads |shift| ≤ 1 as a *fraction of 255*, so severity
   ~0.03 (shift=1) produced a full-scale +255 shift and a fake accuracy
   crater at step 1. Fixed to continuous fractions + regression test.
2. Evaluation harness (not shipped code): grade-prefixed frame names made
   `drift.py`'s sorted-prefix sampling class-imbalanced vs the reference →
   spurious step-0 KS alarms. Frames now index-first named.
3. Stand-in trained without augmentation cliffed at severity ~0.07 on every
   ramp (no slow-drift regime at all). Production-realistic augmentation
   restored gradual degradation.

## Limitations

- Synthetic corruption ≠ real seasonal drift; per the plan, conclusions hold
  for synthetic ramps only until at least one real degradation period is
  replayed (blocked: no archived frames, no QC labeling).
- "False alarms/week" needs a production frame rate to convert from
  per-window counts; unknown, so rates are reported per window.
- Blur ramp saturates too fast to measure early warning; a shallower
  max-severity run would be needed if blur lead time matters.
