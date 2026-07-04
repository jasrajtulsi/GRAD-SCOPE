"""Phase 1 + 2 drift detectors: per-image scores, windowed KS + CUSUM alarms.

Phase 1 (per-image, rolling window): MSP, predictive entropy, GradNorm
(L1 norm of last-layer gradient of KL(uniform || softmax), Huang et al. 2021).
Phase 2 (batch-level): per-sample GradNorm of the *predictive-entropy* loss
(Tent-style, unlabeled), aggregated per batch (mean/median/p10) — GradNorm,
not GSNR, per 2026-07-03 user ruling.

Last-layer gradients use the closed form: for z = W h, dL/dW = g h^T (rank-1),
so ||dL/dW||_1 = ||g||_1 * ||h||_1 with g = dL/dz. No autograd in the hot path.

HARD RULE: no calibration anywhere here. `T` below scales logits *only inside
the gradient computation* (ablation per plan); outputs/logits stay raw.
"""

import numpy as np
import torch
from scipy.stats import ks_2samp


def _softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def msp(logits: np.ndarray) -> np.ndarray:
    return _softmax(logits).max(axis=1)


def pred_entropy(logits: np.ndarray) -> np.ndarray:
    p = _softmax(logits)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def gradnorm(logits: np.ndarray, feats: np.ndarray, loss: str = "uniform", T: float = 1.0) -> np.ndarray:
    """Per-sample L1 GradNorm of `loss` w.r.t. last-layer weights (closed form)."""
    p = _softmax(logits, T)
    if loss == "uniform":  # CE against uniform targets == KL(u || p) + const
        g = p - 1.0 / logits.shape[1]
    elif loss == "entropy":  # Tent loss, unlabeled
        h = -(p * np.log(p + 1e-12)).sum(axis=1, keepdims=True)
        g = -p * (np.log(p + 1e-12) + h)
    elif loss == "pseudo":  # CE against argmax pseudo-labels
        g = p.copy()
        g[np.arange(len(p)), logits.argmax(axis=1)] -= 1.0
    else:
        raise ValueError(loss)
    return np.abs(g / T).sum(axis=1) * np.abs(feats).sum(axis=1)


def gradnorm_lastblock(model: torch.nn.Module, x: torch.Tensor, loss: str = "entropy", T: float = 1.0) -> np.ndarray:
    """Ablation: per-sample L1 GradNorm over last conv block + head (torch.func)."""
    from torch.func import functional_call, grad, vmap

    live = {k: v.detach() for k, v in model.named_parameters() if k.startswith(("body.2", "head"))}
    frozen = {k: v.detach() for k, v in model.named_parameters() if not k.startswith(("body.2", "head"))}
    buffers = dict(model.named_buffers())

    def loss_fn(params: dict, xi: torch.Tensor) -> torch.Tensor:
        logits = functional_call(model, ({**frozen, **params}, buffers), (xi.unsqueeze(0),))[0]
        logp = (logits / T).log_softmax(dim=1)
        p = logp.exp()
        if loss == "uniform":
            return -logp.mean(dim=1).sum()
        if loss == "entropy":
            return -(p * logp).sum()
        return torch.nn.functional.cross_entropy(logits / T, logits.argmax(dim=1))

    grads = vmap(grad(loss_fn), in_dims=(None, 0))(live, x)
    total = sum(g.abs().flatten(start_dim=1).sum(dim=1) for g in grads.values())
    return total.cpu().numpy()


class WindowDetector:
    """Unified alarm interface (Phase 1 rolling windows AND Phase 2 batches).

    Against a fixed clean reference: KS test on each window's raw scores
    (alarm after `persist` consecutive p < alpha) + two-sided CUSUM on the
    window aggregate, standardized by bootstrap over the reference.
    """

    def __init__(self, reference: np.ndarray, window: int, agg: str = "mean",
                 alpha: float = 0.01, persist: int = 2, k: float = 0.5, h: float = 8.0):
        self.ref = np.asarray(reference)
        self.aggfn = {"mean": np.mean, "median": np.median, "p10": lambda a: np.percentile(a, 10)}[agg]
        rng = np.random.default_rng(0)
        boots = [self.aggfn(rng.choice(self.ref, size=window)) for _ in range(200)]
        self.mu, self.sigma = float(np.mean(boots)), float(np.std(boots) + 1e-12)
        self.alpha, self.persist, self.k, self.h = alpha, persist, k, h
        self.ks_run = 0
        self.sp = self.sn = 0.0

    def update(self, window_scores: np.ndarray) -> dict:
        p = ks_2samp(window_scores, self.ref).pvalue
        self.ks_run = self.ks_run + 1 if p < self.alpha else 0
        z = (self.aggfn(window_scores) - self.mu) / self.sigma
        self.sp = max(0.0, self.sp + z - self.k)
        self.sn = max(0.0, self.sn - z - self.k)
        return {"ks_p": p, "z": z,
                "alarm_ks": self.ks_run >= self.persist,
                "alarm_cusum": self.sp > self.h or self.sn > self.h}


def first_alarm(flags: list[bool]) -> int | None:
    return next((i for i, a in enumerate(flags) if a), None)
