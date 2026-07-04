"""Stand-in grading model + synthetic fruit frames (no production model exists).

Procedural "fruit on conveyor" images with 3 quality grades driven by blemish
count (A=0, B=1-3, C=5-9). A small CNN trained on clean frames stands in for
the production grader so Phase 1/2 detectors have a model to instrument.

Usage:
    python standin.py --train                      # trains, saves standin.pt
    python standin.py --dump-frames DIR --n 400    # labeled clean frames for drift.py
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG = 96
GRADES = 3
_BLEMISH_COUNT = {0: (0, 0), 1: (1, 3), 2: (5, 9)}


def make_frame(grade: int, rng: np.random.Generator) -> np.ndarray:
    """One BGR uint8 conveyor frame of a fruit with grade-dependent blemishes."""
    img = np.full((IMG, IMG, 3), 90, np.uint8)  # belt
    img = cv2.add(img, rng.integers(0, 25, (IMG, IMG, 3), dtype=np.uint8))
    cx, cy = rng.integers(38, 58, 2)
    ax, ay = rng.integers(24, 34, 2)
    # ripeness varies but does NOT determine grade (grade = blemishes only)
    color = (int(rng.integers(20, 60)), int(rng.integers(120, 200)), int(rng.integers(150, 230)))
    cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), int(rng.integers(0, 180)), 0, 360, color, -1)
    lo, hi = _BLEMISH_COUNT[grade]
    for _ in range(int(rng.integers(lo, hi + 1))):
        r = int(rng.integers(2, 6))
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 0.7)
        bx = int(cx + rad * ax * np.cos(ang))
        by = int(cy + rad * ay * np.sin(ang))
        cv2.circle(img, (bx, by), r, (15, 25, 35), -1)
    return img


def make_batch(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, GRADES, n)
    imgs = np.stack([make_frame(int(g), rng) for g in labels])
    return imgs, labels


def to_tensor(imgs_bgr: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(imgs_bgr[..., ::-1].copy()).float().div(255).permute(0, 3, 1, 2)
    return x.to(device)


class StandIn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.body = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, GRADES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, penultimate features) — features feed GradNorm closed form."""
        h = self.body(x)
        return self.head(h), h


def train(device: str = "cuda", seed: int = 0, path: str = "standin.pt") -> float:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StandIn().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for _ in range(300):
        imgs, labels = make_batch(256, rng)
        x = to_tensor(imgs, device)
        # production-realistic augmentation: without it the CNN cliffs at tiny
        # severities and every ramp fails at step ~2 (no slow-drift regime at all)
        gain = 1 + (torch.rand(len(x), 1, 1, 1, device=device) - 0.5) * 0.4
        offs = (torch.rand(len(x), 1, 1, 1, device=device) - 0.5) * 0.16
        chan = (torch.rand(len(x), 3, 1, 1, device=device) - 0.5) * 0.06
        x = x * gain + offs + chan + torch.randn_like(x) * 0.03
        if torch.rand(()) < 0.3:
            x = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
        x = x.clamp(0, 1)
        loss = nn.functional.cross_entropy(model(x)[0], torch.from_numpy(labels).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    imgs, labels = make_batch(1000, rng)
    with torch.no_grad():
        acc = (model(to_tensor(imgs, device))[0].argmax(1).cpu().numpy() == labels).mean()
    torch.save(model.state_dict(), path)
    return float(acc)


def load(path: str = "standin.pt", device: str = "cuda") -> StandIn:
    model = StandIn().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def dump_frames(out: Path, n: int, seed: int) -> None:
    """Clean labeled frames for drift.py; label parseable from filename."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        g = i % GRADES
        # index-first name: any sorted prefix stays grade-balanced (drift.py
        # draws the first frames_per_step frames in sorted order)
        cv2.imwrite(str(out / f"{i:05d}_grade{g}.png"), make_frame(g, rng))


def label_of(frame_name: str) -> int:
    return int(re.search(r"grade(\d)", Path(frame_name).name).group(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dump-frames", type=Path)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.train:
        print(f"clean accuracy: {train(seed=args.seed):.3f}")
    if args.dump_frames:
        dump_frames(args.dump_frames, args.n, args.seed)
        print(f"wrote {args.n} frames to {args.dump_frames}")
