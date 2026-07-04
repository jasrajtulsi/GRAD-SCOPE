"""Phase 3: supervised fragility gate for candidate models.

Before a retrained candidate ships, fine-tune it briefly on the weekly
labeled QC batch (backbone frozen, last block + head only) and record the
per-parameter GSNR of the last-layer gradients (batch mean^2 / variance,
arXiv 2001.07384: low GSNR <-> large generalization gap -> fragile).
GradNorm is logged alongside from the same per-sample gradients; the gate
metric is configurable (`gate_config.json`), default GSNR per the plan's
Phase 3 — the 2026-07-03 GradNorm ruling covered the unlabeled detection
path, and this supervised gate is where GSNR has direct literature support.

Cold start (plan): no absolute threshold until 5+ cycles — gate is relative:
BLOCK if candidate metric < rel_margin * incumbent metric, or candidate QC
accuracy < incumbent - acc_margin. History appends to
results/gate_history.csv; absolute_threshold in the config stays null until
enough cycles accumulate.

No real QC labeling process exists (2026-07-03) — QC batches are synthetic
labeled frames until one does.

Usage:
    python gate.py --candidate cand.pt --incumbent standin.pt
    python gate.py --demo    # builds healthy + fragile candidates, gates them
"""

import argparse
import copy
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

import standin

CONFIG = Path("gate_config.json")
HISTORY = Path("results") / "gate_history.csv"
EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def per_sample_stats(logits: np.ndarray, feats: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(GSNR, GradNorm) of last-layer weight gradients under labeled CE.

    Per-sample grad of CE w.r.t. last-layer W is rank-1: (p - onehot) x h,
    so per-parameter per-sample grads come from an einsum, no autograd.
    GSNR: per-parameter mean^2/var over the batch, averaged over parameters.
    """
    g = _softmax(logits)
    g[np.arange(len(g)), labels] -= 1.0
    G = np.einsum("ic,id->icd", g, feats)
    gsnr = float((G.mean(0) ** 2 / (G.var(0) + EPS)).mean())
    gradnorm = float((np.abs(g).sum(1) * np.abs(feats).sum(1)).mean())
    return gsnr, gradnorm


def finetune_and_measure(model: standin.StandIn, imgs: np.ndarray, labels: np.ndarray,
                         device: str, steps: int = 100, lr: float = 1e-4, batch: int = 64) -> dict:
    """Fine-tune a copy on the QC batch (last block + head), record GSNR/GradNorm per step."""
    m = copy.deepcopy(model).to(device).train()
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(("body.2", "head"))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr)

    # QC accuracy of the candidate as shipped (before any fine-tuning)
    model.eval()
    with torch.no_grad():
        qc_acc = float((model(standin.to_tensor(imgs, device))[0].argmax(1).cpu().numpy() == labels).mean())

    rng = np.random.default_rng(0)
    gsnrs, gradnorms = [], []
    y = torch.from_numpy(labels).to(device)
    for _ in range(steps):
        idx = rng.choice(len(imgs), size=min(batch, len(imgs)), replace=False)
        x = standin.to_tensor(imgs[idx], device)
        logits, feats = m(x)
        gs, gn = per_sample_stats(logits.detach().cpu().numpy(), feats.detach().cpu().numpy(), labels[idx])
        gsnrs.append(gs)
        gradnorms.append(gn)
        loss = torch.nn.functional.cross_entropy(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"gsnr": float(np.mean(gsnrs)), "gradnorm": float(np.mean(gradnorms)),
            "qc_acc": qc_acc, "gsnr_trajectory": gsnrs}


def load_config() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    cfg = {"metric": "gsnr", "rel_margin": 0.8, "acc_margin": 0.02, "absolute_threshold": None,
           "note": "relative gate until >=5 cycles accumulate, then set absolute_threshold"}
    CONFIG.write_text(json.dumps(cfg, indent=2))
    return cfg


def gate(cand: dict, inc: dict, cfg: dict) -> tuple[str, list[str]]:
    reasons = []
    m = cfg["metric"]
    if cfg.get("absolute_threshold") is not None and cand[m] < cfg["absolute_threshold"]:
        reasons.append(f"{m} {cand[m]:.4g} < absolute threshold {cfg['absolute_threshold']:.4g}")
    if cand[m] < cfg["rel_margin"] * inc[m]:
        reasons.append(f"{m} {cand[m]:.4g} < {cfg['rel_margin']} x incumbent {inc[m]:.4g}")
    if cand["qc_acc"] < inc["qc_acc"] - cfg["acc_margin"]:
        reasons.append(f"qc_acc {cand['qc_acc']:.3f} < incumbent {inc['qc_acc']:.3f} - {cfg['acc_margin']}")
    return ("BLOCK" if reasons else "PASS"), reasons


def append_history(candidate: str, incumbent: str, cand: dict, inc: dict, verdict: str) -> None:
    HISTORY.parent.mkdir(exist_ok=True)
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "candidate", "incumbent", "cand_gsnr", "cand_gradnorm",
                        "cand_qc_acc", "inc_gsnr", "inc_gradnorm", "inc_qc_acc", "verdict"])
        w.writerow([date.today().isoformat(), candidate, incumbent,
                    f"{cand['gsnr']:.6g}", f"{cand['gradnorm']:.6g}", f"{cand['qc_acc']:.4f}",
                    f"{inc['gsnr']:.6g}", f"{inc['gradnorm']:.6g}", f"{inc['qc_acc']:.4f}", verdict])


def run_gate(candidate_path: str, incumbent_path: str, device: str, qc_seed: int = 7, qc_n: int = 512) -> str:
    imgs, labels = standin.make_batch(qc_n, np.random.default_rng(qc_seed))
    cfg = load_config()
    cand = finetune_and_measure(standin.load(candidate_path, device), imgs, labels, device)
    inc = finetune_and_measure(standin.load(incumbent_path, device), imgs, labels, device)
    verdict, reasons = gate(cand, inc, cfg)
    append_history(candidate_path, incumbent_path, cand, inc, verdict)
    print(f"{verdict}: {candidate_path} vs {incumbent_path}")
    print(f"  candidate: gsnr={cand['gsnr']:.4g} gradnorm={cand['gradnorm']:.4g} qc_acc={cand['qc_acc']:.3f}")
    print(f"  incumbent: gsnr={inc['gsnr']:.4g} gradnorm={inc['gradnorm']:.4g} qc_acc={inc['qc_acc']:.3f}")
    for r in reasons:
        print(f"  - {r}")
    return verdict


def _demo(device: str) -> None:
    """Gate three synthetic candidates: healthy retrain, overfit, undertrained."""
    torch.manual_seed(1)
    print("building demo candidates...")
    # healthy: the realistic weekly retrain — continue training the incumbent
    # on fresh augmented data (fresh from-scratch seeds are too unstable at
    # 300 steps to represent a "healthy" candidate)
    m = standin.load(device=device).train()
    opt = torch.optim.Adam(m.parameters(), 5e-4)
    rng = np.random.default_rng(5)
    for _ in range(100):
        imgs, labels = standin.make_batch(256, rng)
        x = standin.to_tensor(imgs, device)
        gain = 1 + (torch.rand(len(x), 1, 1, 1, device=device) - 0.5) * 0.4
        x = (x * gain + torch.randn_like(x) * 0.03).clamp(0, 1)
        loss = torch.nn.functional.cross_entropy(m(x)[0], torch.from_numpy(labels).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(m.state_dict(), "cand_healthy.pt")
    print("  healthy candidate: incumbent + 100 augmented continue-train steps")

    # overfit: memorize 256 fixed frames, no augmentation (accuracy can stay
    # decent while generalization degrades — the case the GSNR leg is for)
    m = standin.load(device=device).train()
    imgs, labels = standin.make_batch(256, np.random.default_rng(9))
    x, y = standin.to_tensor(imgs, device), torch.from_numpy(labels).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(500):
        loss = torch.nn.functional.cross_entropy(m(x)[0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(m.state_dict(), "cand_overfit.pt")
    print("  overfit candidate: 500 steps on 256 fixed frames")

    # undertrained: fresh model, 15 training steps only
    torch.manual_seed(2)
    m = standin.StandIn().to(device).train()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    rng = np.random.default_rng(2)
    for _ in range(15):
        imgs, labels = standin.make_batch(256, rng)
        loss = torch.nn.functional.cross_entropy(m(standin.to_tensor(imgs, device))[0],
                                                 torch.from_numpy(labels).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(m.state_dict(), "cand_undertrained.pt")
    print("  undertrained candidate: 15 steps")

    for cand in ("cand_healthy.pt", "cand_overfit.pt", "cand_undertrained.pt"):
        run_gate(cand, "standin.pt", device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate")
    ap.add_argument("--incumbent", default="standin.pt")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.demo:
        _demo(dev)
    elif args.candidate:
        run_gate(args.candidate, args.incumbent, dev)
    else:
        ap.error("need --candidate or --demo")
