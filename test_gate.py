"""Self-check for gate.py. Run: python test_gate.py"""

import numpy as np
import torch

from gate import gate, per_sample_stats


def check_stats_closed_form() -> None:
    """einsum per-sample grads must match autograd per-sample CE grads."""
    torch.manual_seed(0)
    W = torch.randn(3, 8, requires_grad=True)
    h = torch.randn(6, 8)
    labels = np.array([0, 1, 2, 0, 1, 2])
    logits = h @ W.T
    grads = []
    for i in range(len(h)):
        li = torch.nn.functional.cross_entropy((h[i : i + 1] @ W.T), torch.tensor([labels[i]]))
        (g,) = torch.autograd.grad(li, W)
        grads.append(g.numpy().copy())
    G = np.stack(grads)
    want_gsnr = (G.mean(0) ** 2 / (G.var(0) + 1e-12)).mean()
    want_gn = np.abs(G).sum(axis=(1, 2)).mean()
    got_gsnr, got_gn = per_sample_stats(logits.detach().numpy(), h.numpy(), labels)
    assert abs(got_gsnr - want_gsnr) < 1e-4 * max(1, want_gsnr), (got_gsnr, want_gsnr)
    assert abs(got_gn - want_gn) < 1e-4 * max(1, want_gn), (got_gn, want_gn)


def check_gate_rule() -> None:
    cfg = {"metric": "gsnr", "rel_margin": 0.8, "acc_margin": 0.02, "absolute_threshold": None}
    inc = {"gsnr": 1.0, "qc_acc": 0.95}
    assert gate({"gsnr": 0.9, "qc_acc": 0.95}, inc, cfg)[0] == "PASS"
    assert gate({"gsnr": 0.5, "qc_acc": 0.95}, inc, cfg)[0] == "BLOCK"  # metric leg
    assert gate({"gsnr": 0.9, "qc_acc": 0.90}, inc, cfg)[0] == "BLOCK"  # accuracy leg
    cfg["absolute_threshold"] = 2.0
    assert gate({"gsnr": 1.5, "qc_acc": 0.95}, inc, cfg)[0] == "BLOCK"  # absolute leg


if __name__ == "__main__":
    check_stats_closed_form()
    check_gate_rule()
    print("all checks passed")
