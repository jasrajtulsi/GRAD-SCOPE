"""Self-check for detectors.py. Run: python test_detectors.py"""

import numpy as np
import torch

from detectors import WindowDetector, first_alarm, gradnorm, gradnorm_lastblock, msp, pred_entropy
from standin import StandIn


def check_closed_form() -> None:
    """Closed-form last-layer GradNorm must equal autograd on a linear head."""
    torch.manual_seed(0)
    W = torch.randn(4, 8, requires_grad=True)
    h = torch.randn(5, 8)
    z = h @ W.T
    for loss_name, T in [("uniform", 1.0), ("entropy", 1.0), ("pseudo", 1.0), ("entropy", 2.0)]:
        got = gradnorm(z.detach().numpy(), h.numpy(), loss=loss_name, T=T)
        for i in range(len(h)):
            zi = (h[i : i + 1] @ W.T) / T
            logp = zi.log_softmax(1)
            p = logp.exp()
            if loss_name == "uniform":
                li = -logp.mean(1).sum()
            elif loss_name == "entropy":
                li = -(p * logp).sum()
            else:
                li = torch.nn.functional.cross_entropy(zi, zi.argmax(1))
            (g,) = torch.autograd.grad(li, W, retain_graph=False)
            assert abs(g.abs().sum().item() - got[i]) < 1e-4, (loss_name, T, i)


def check_detector() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 3000)
    det = WindowDetector(ref, window=256)
    for _ in range(20):  # clean stream: silence
        r = det.update(rng.normal(0, 1, 256))
        assert not r["alarm_ks"] and not r["alarm_cusum"], "false alarm on clean stream"
    flags = []
    for i in range(20):  # slow ramp: must alarm
        r = det.update(rng.normal(0.15 * i, 1, 256))
        flags.append(r["alarm_ks"] or r["alarm_cusum"])
    assert first_alarm(flags) is not None, "no alarm under drift"


def check_scores_and_lastblock() -> None:
    logits = np.zeros((3, 4))  # uniform predictions
    assert np.allclose(msp(logits), 0.25) and np.allclose(pred_entropy(logits), np.log(4))
    torch.manual_seed(0)
    model = StandIn().eval()
    g = gradnorm_lastblock(model, torch.rand(6, 3, 96, 96))
    assert g.shape == (6,) and np.isfinite(g).all()


if __name__ == "__main__":
    check_closed_form()
    check_detector()
    check_scores_and_lastblock()
    print("all checks passed")
