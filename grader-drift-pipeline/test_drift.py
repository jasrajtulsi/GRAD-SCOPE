"""Self-check for drift.py. Run: python test_drift.py"""

import csv
import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

from drift import IMG_EXTS, generate, make_transform


def _fake_frames(d: Path, n: int = 3) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{i}.png"), img)


def _run(out: Path, corruption: str, seed: int = 42) -> list[dict]:
    with tempfile.TemporaryDirectory() as src:
        src = Path(src)
        _fake_frames(src)
        manifest = generate(src, out, corruption, steps=4, max_severity=1.0, seed=seed)
        with manifest.open() as f:
            return list(csv.DictReader(f))


def _dir_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*.png")):
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    for corruption in ["lighting", "blur", "dust", "colortemp"]:
        with tempfile.TemporaryDirectory() as out:
            out = Path(out)
            rows = _run(out, corruption)
            # 5 steps (0..4) x 3 frames
            assert len(rows) == 15, (corruption, len(rows))
            sevs = sorted({float(r["severity"]) for r in rows})
            assert sevs[0] == 0.0 and sevs[-1] == 1.0, sevs  # ramp spans clean -> max
            for r in rows:
                assert (out / r["output_frame"]).is_file(), r["output_frame"]
            # step 0 frames are the clean baseline
            clean = [r for r in rows if r["step"] == "0"]
            assert all(float(r["severity"]) == 0.0 for r in clean)
            # corrupted frames actually differ from clean ones
            c0 = cv2.imread(str(out / clean[0]["output_frame"]))
            last = [r for r in rows if r["step"] == "4" and r["source_frame"] == clean[0]["source_frame"]]
            c4 = cv2.imread(str(out / last[0]["output_frame"]))
            assert not np.array_equal(c0, c4), f"{corruption}: max severity changed nothing"

    # reproducibility: same seed -> byte-identical outputs
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _run(Path(a), "dust")
        _run(Path(b), "dust")
        assert _dir_hash(Path(a)) == _dir_hash(Path(b)), "same seed must reproduce identical bytes"

    assert ".png" in IMG_EXTS
    make_transform("lighting", 0.5)  # smoke: mid-severity constructs

    # regression: low-severity colortemp must be a small shift, not a
    # full-scale one (albumentations reads |shift| <= 1 as fraction of 255)
    img = np.full((8, 8, 3), 100, np.uint8)
    t = make_transform("colortemp", 0.03)
    t.set_random_seed(0)
    diff = np.abs(t(image=img)["image"].astype(int) - 100).max()
    assert diff <= 5, f"low-severity colortemp shifted by {diff}"
    print("all checks passed")


if __name__ == "__main__":
    main()
