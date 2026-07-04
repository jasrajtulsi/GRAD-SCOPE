"""Phase 1 + 2 end-to-end evaluation on synthetic slow-drift ramps.

Trains the stand-in grader (if needed), generates 4 corruption ramps with
drift.py, streams them through the model, runs Phase 1 windowed baselines
(MSP / entropy / uniform-GradNorm) and the Phase 2 batch-level
entropy-GradNorm detector, and scores everything against the accuracy
ground-truth curve. Also runs the Phase 2 ablations (loss / T / aggregation /
batch size / parameter subset) on the dust ramp, and a clean-only stream for
false-alarm rates.

Outputs: results/steps_{corruption}.csv, results/summary.json.
Run: .venv\\Scripts\\python.exe evaluate.py
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import drift
import standin
from detectors import WindowDetector, gradnorm, gradnorm_lastblock, msp, pred_entropy

STEPS = 30
PER_STEP = 256          # frames per time step == Phase 2 batch size (plan: >=256)
WINDOW, STRIDE = 512, 256  # Phase 1 rolling window (plan: 500-2000)
ACC_DROP = 0.05         # "business threshold": clean accuracy minus 5 points
CORRUPTIONS = ["lighting", "blur", "dust", "colortemp"]
DATA, RESULTS = Path("data"), Path("results")


@torch.no_grad()
def infer(model, imgs: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    ls, fs = [], []
    for i in range(0, len(imgs), 512):
        logits, feats = model(standin.to_tensor(imgs[i : i + 512], device))
        ls.append(logits.cpu().numpy())
        fs.append(feats.cpu().numpy())
    return np.concatenate(ls), np.concatenate(fs)


def lastblock_scores(model, imgs: np.ndarray, device: str) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), 512):
        out.append(gradnorm_lastblock(model, standin.to_tensor(imgs[i : i + 512], device)))
    return np.concatenate(out)


def load_ramp(corr: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Per step: (imgs BGR uint8, labels, severity) from drift.py output."""
    root = DATA / f"ramp_{corr}"
    with (root / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append(r)
    out = []
    for s in sorted(by_step):
        rs = by_step[s]
        imgs = np.stack([cv2.imread(str(root / r["output_frame"])) for r in rs])
        labels = np.array([standin.label_of(r["source_frame"]) for r in rs])
        out.append((imgs, labels, float(rs[0]["severity"])))
    return out


def fail_step(accs: list[float], clean_acc: float) -> int | None:
    """First of >=2 consecutive steps below clean_acc - ACC_DROP."""
    thr = clean_acc - ACC_DROP
    for s in range(len(accs) - 1):
        if accs[s] < thr and accs[s + 1] < thr:
            return s
    return len(accs) - 1 if accs and accs[-1] < thr else None


def rolling_alarm_step(scores: np.ndarray, ref: np.ndarray) -> dict:
    """Phase 1: first alarm step for KS and CUSUM over rolling windows."""
    det = WindowDetector(ref, window=WINDOW)
    first = {"ks": None, "cusum": None}
    for i in range(0, len(scores) - WINDOW + 1, STRIDE):
        r = det.update(scores[i : i + WINDOW])
        step = (i + WINDOW - 1) // PER_STEP
        for rule in ("ks", "cusum"):
            if first[rule] is None and r[f"alarm_{rule}"]:
                first[rule] = step
    return first


def batch_alarm_step(scores: np.ndarray, ref: np.ndarray, batch: int = PER_STEP, agg: str = "mean") -> dict:
    """Phase 2: first alarm step over consecutive batches."""
    det = WindowDetector(ref, window=batch, agg=agg)
    first = {"ks": None, "cusum": None}
    for i in range(len(scores) // batch):
        r = det.update(scores[i * batch : (i + 1) * batch])
        step = ((i + 1) * batch - 1) // PER_STEP
        for rule in ("ks", "cusum"):
            if first[rule] is None and r[f"alarm_{rule}"]:
                first[rule] = step
    return first


def _abl_entry(scores, ref, clean_scores, batch: int = PER_STEP, agg: str = "mean") -> dict:
    alarm = batch_alarm_step(scores, ref, batch=batch, agg=agg)
    det = WindowDetector(ref, window=batch, agg=agg)
    n_fa = 0
    for i in range(len(clean_scores) // batch):
        r = det.update(clean_scores[i * batch : (i + 1) * batch])
        n_fa += r["alarm_ks"] or r["alarm_cusum"]
    return {"alarm": alarm, "clean_fa_windows": int(n_fa)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)

    if not Path("standin.pt").exists():
        acc = standin.train(device=device)
        print(f"trained stand-in, clean accuracy {acc:.3f}")
    model = standin.load(device=device)

    pool = DATA / "clean_pool"
    if not pool.exists():
        standin.dump_frames(pool, 400, seed=1)
    for corr in CORRUPTIONS:
        if not (DATA / f"ramp_{corr}" / "manifest.csv").exists():
            drift.generate(pool, DATA / f"ramp_{corr}", corr, STEPS, 1.0, 42, PER_STEP)
            print(f"generated ramp_{corr}")

    # clean reference (deployment-start baseline) and clean stream (false alarms)
    ref_imgs, _ = standin.make_batch(2560, np.random.default_rng(2))
    ref_l, ref_f = infer(model, ref_imgs, device)
    clean_imgs, _ = standin.make_batch(STEPS * PER_STEP, np.random.default_rng(3))
    cl_l, cl_f = infer(model, clean_imgs, device)

    def signals(l, f):
        return {"msp": msp(l), "entropy": pred_entropy(l), "gradnorm": gradnorm(l, f, "uniform"),
                "egn": gradnorm(l, f, "entropy")}

    ref_s, cl_s = signals(ref_l, ref_f), signals(cl_l, cl_f)
    summary: dict = {"config": {"steps": STEPS, "per_step": PER_STEP, "window": WINDOW,
                                "stride": STRIDE, "acc_drop": ACC_DROP}}

    # ---- ramps: truth curve + alarms -------------------------------------
    for corr in CORRUPTIONS:
        steps = load_ramp(corr)
        per_image = {k: [] for k in ref_s}
        accs, sevs = [], []
        for imgs, labels, sev in steps:
            l, f = infer(model, imgs, device)
            for k, v in signals(l, f).items():
                per_image[k].append(v)
            accs.append(float((l.argmax(1) == labels).mean()))
            sevs.append(sev)
        stream = {k: np.concatenate(v) for k, v in per_image.items()}

        clean_acc = accs[0]
        fs = fail_step(accs, clean_acc)
        entry = {"clean_acc": clean_acc, "fail_step": fs, "accs": accs, "severities": sevs, "alarms": {}}
        for sig in ("msp", "entropy", "gradnorm"):  # Phase 1
            entry["alarms"][sig] = rolling_alarm_step(stream[sig], ref_s[sig])
        entry["alarms"]["egn_batch"] = batch_alarm_step(stream["egn"], ref_s["egn"])  # Phase 2
        summary[corr] = entry

        with (RESULTS / f"steps_{corr}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "severity", "acc", "msp_mean", "entropy_mean", "gradnorm_mean", "egn_mean"])
            for s in range(len(accs)):
                w.writerow([s, f"{sevs[s]:.4f}", f"{accs[s]:.4f}"] +
                           [f"{per_image[k][s].mean():.5f}" for k in ("msp", "entropy", "gradnorm", "egn")])
        print(f"{corr}: clean_acc={clean_acc:.3f} fail_step={fs} alarms={entry['alarms']}")

    # ---- false alarms on the clean stream --------------------------------
    fa: dict = {}
    for sig in ("msp", "entropy", "gradnorm"):
        det = WindowDetector(ref_s[sig], window=WINDOW)
        n = 0
        for i in range(0, len(cl_s[sig]) - WINDOW + 1, STRIDE):
            r = det.update(cl_s[sig][i : i + WINDOW])
            n += r["alarm_ks"] or r["alarm_cusum"]
        fa[sig] = int(n)
    fa["egn_batch"] = _abl_entry(cl_s["egn"], ref_s["egn"], cl_s["egn"])["clean_fa_windows"]
    summary["clean_false_alarm_windows"] = fa
    print(f"clean-stream false-alarm windows: {fa}")

    # ---- Phase 2 ablations on the dust ramp ------------------------------
    dust_imgs = np.concatenate([s[0] for s in load_ramp("dust")])
    dust_l, dust_f = infer(model, dust_imgs, device)
    abl: dict = {}
    for name, loss, T in [("entropy_T1", "entropy", 1.0), ("uniform_T1", "uniform", 1.0),
                          ("pseudo_T1", "pseudo", 1.0), ("entropy_T2", "entropy", 2.0),
                          ("entropy_T10", "entropy", 10.0)]:
        abl[f"loss:{name}"] = _abl_entry(gradnorm(dust_l, dust_f, loss, T),
                                         gradnorm(ref_l, ref_f, loss, T),
                                         gradnorm(cl_l, cl_f, loss, T))
    for agg in ("median", "p10"):
        abl[f"agg:{agg}"] = _abl_entry(gradnorm(dust_l, dust_f, "entropy"), ref_s["egn"], cl_s["egn"], agg=agg)
    for b in (128, 512):
        abl[f"batch:{b}"] = _abl_entry(gradnorm(dust_l, dust_f, "entropy"), ref_s["egn"], cl_s["egn"], batch=b)
    abl["params:lastblock"] = _abl_entry(lastblock_scores(model, dust_imgs, device),
                                         lastblock_scores(model, ref_imgs, device),
                                         lastblock_scores(model, clean_imgs, device))
    summary["ablation_dust"] = abl
    for k, v in abl.items():
        print(f"ablation {k}: {v}")

    with (RESULTS / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
