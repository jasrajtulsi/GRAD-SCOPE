"""Phase 0: synthetic slow-drift corruption generator.

Applies ImageNet-C-style corruptions to a folder of production frames with
intensity ramping linearly over a simulated timeline (slow drift, not sudden
failure). Emits corrupted frames grouped by time step plus a manifest.csv
ground-truth timeline (frame, step, corruption, severity) for later detector
evaluation (windowed MSP/entropy, GradNorm).

Usage:
    python drift.py --input frames/ --output out/ --corruption dust \
        --steps 50 --max-severity 1.0 --seed 42

Corruptions (all severity in [0,1], continuous — imagecorruptions' discrete
1-5 severities are too coarse for slow ramps, hence albumentations):
    lighting   brightness/gamma shift (lamp aging, ambient change)
    blur       motion blur (belt speed / vibration)
    dust       gaussian noise + dark specks (lens dust, particles)
    colortemp  channel shift toward warm (lamp color drift)
"""

import argparse
import csv
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def make_transform(corruption: str, severity: float) -> A.BasicTransform:
    """Severity in [0,1] -> albumentations transform at that intensity, p=1."""
    s = float(np.clip(severity, 0.0, 1.0))
    if corruption == "lighting":
        # up to -60% brightness, -30% contrast at full severity
        return A.RandomBrightnessContrast(
            brightness_limit=(-0.6 * s, -0.6 * s), contrast_limit=(-0.3 * s, -0.3 * s), p=1.0
        )
    if corruption == "blur":
        # kernel must be odd and >=3; ramp 3 -> 25
        k = max(3, int(3 + 22 * s) | 1)
        return A.MotionBlur(blur_limit=(k, k), p=1.0)
    if corruption == "dust":
        return A.Compose(
            [
                A.GaussNoise(std_range=(0.02 + 0.18 * s, 0.02 + 0.18 * s), p=1.0),
                # dark specks: up to 40 holes of ~2% image size at full severity
                A.CoarseDropout(
                    num_holes_range=(1, max(1, int(40 * s))),
                    hole_height_range=(0.005, 0.02),
                    hole_width_range=(0.005, 0.02),
                    fill=0,
                    p=1.0 if s > 0 else 0.0,
                ),
            ]
        )
    if corruption == "colortemp":
        # warm drift: +R, -B, up to 50/255 at full severity. Pass fractions:
        # albumentations reads |shift| <= 1 as a fraction of 255, so an int
        # shift of 1 (severity ~0.02-0.04) would mean a full-scale +255 shift.
        shift = 50 * s / 255
        return A.RGBShift(
            r_shift_limit=(shift, shift), g_shift_limit=(0, 0), b_shift_limit=(-shift, -shift), p=1.0
        )
    raise ValueError(f"unknown corruption: {corruption}")


def generate(
    input_dir: Path,
    output_dir: Path,
    corruption: str,
    steps: int,
    max_severity: float,
    seed: int,
    frames_per_step: int | None = None,
) -> Path:
    """Ramp corruption over `steps` time steps; returns manifest path.

    Each step draws `frames_per_step` frames from input_dir and applies the
    corruption at severity (step/steps)*max_severity. Step 0 is always clean
    (severity 0) — the deployment-time baseline.
    """
    frames = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no images found in {input_dir}")
    frames_per_step = frames_per_step or len(frames)

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "severity", "corruption", "source_frame", "output_frame", "seed"])
        for step in range(steps + 1):
            severity = max_severity * step / steps
            step_dir = output_dir / f"step_{step:04d}"
            step_dir.mkdir(exist_ok=True)
            transform = make_transform(corruption, severity)
            for i in range(frames_per_step):
                src = frames[i % len(frames)] if frames_per_step <= len(frames) else rng.choice(frames)
                img = cv2.imread(str(src))
                if img is None:
                    raise SystemExit(f"unreadable image: {src}")
                # per-frame deterministic seed so runs are reproducible
                # (albumentations 2.x has internal RNG; global seeds do nothing)
                frame_seed = seed * 1_000_003 + step * 10_007 + i
                transform.set_random_seed(frame_seed)
                out = transform(image=img)["image"] if severity > 0 else img
                out_name = f"{src.stem}_s{step:04d}_{i:04d}.png"
                cv2.imwrite(str(step_dir / out_name), out)
                writer.writerow(
                    [step, f"{severity:.4f}", corruption, src.name, f"step_{step:04d}/{out_name}", frame_seed]
                )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True, help="dir of clean production frames")
    ap.add_argument("--output", type=Path, required=True, help="output dir for ramped dataset")
    ap.add_argument("--corruption", choices=["lighting", "blur", "dust", "colortemp"], required=True)
    ap.add_argument("--steps", type=int, default=50, help="ramp length in time steps (step 0 = clean)")
    ap.add_argument("--max-severity", type=float, default=1.0, help="severity at final step, in (0,1]")
    ap.add_argument("--frames-per-step", type=int, default=None, help="frames per step (default: all inputs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    manifest = generate(
        args.input, args.output, args.corruption, args.steps, args.max_severity, args.seed, args.frames_per_step
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
