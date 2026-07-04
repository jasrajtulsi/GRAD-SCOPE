"""Command-line entry point for GRAD-SCOPE experiments.

Parses arguments, builds a BaselineConfig, and dispatches to the selected
failure-mode experiment:

* ``--exp 1`` — pathologically high learning rate (lr = 1.0).
* ``--exp 2`` — BatchNorm removed (``remove_bn=True``).
* ``--exp 3`` — pathological constant initialization (``bad_init()``).

Every experiment attaches a GradientLogger (live_state.json is rewritten each
epoch), saves the full GSNR history to ``results/<run_name>_gsnr.csv``, computes
early-warning lead times at the end, and prints a results table.

Usage:
    python -m experiments.run_experiment --exp 1 --arch resnet20
"""

from __future__ import annotations

import argparse

from experiments.baseline import BaselineConfig, run_baseline

# Per-experiment overrides applied on top of the baseline configuration.
EXPERIMENTS: dict[int, dict] = {
    1: {"label": "high learning rate (lr=1.0)", "overrides": {"lr": 1.0}},
    2: {"label": "no BatchNorm", "overrides": {"remove_bn": True}},
    3: {"label": "bad initialization (all params = 0.001)", "overrides": {"bad_init": True}},
}

EXPERIMENT_NAMES: dict[int, str] = {1: "exp1_high_lr", 2: "exp2_no_bn", 3: "exp3_bad_init"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments into a namespace."""
    parser = argparse.ArgumentParser(description="Run a GRAD-SCOPE failure-mode experiment.")
    parser.add_argument(
        "--exp",
        type=int,
        required=True,
        choices=sorted(EXPERIMENTS),
        help="experiment number: 1=high lr, 2=no BatchNorm, 3=bad init",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet20",
        choices=["resnet20", "vgg11"],
        help="model architecture (default: resnet20)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size (default: 128)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BaselineConfig:
    """Build the experiment's config from CLI args plus its pathology overrides."""
    config = BaselineConfig(
        model=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        run_name=f"{EXPERIMENT_NAMES[args.exp]}_{args.arch}",
    )
    for key, value in EXPERIMENTS[args.exp]["overrides"].items():
        setattr(config, key, value)
    return config


def print_lead_time_table(lead_times: dict) -> None:
    """Print each early-warning signal's firing epoch and lead time over failure."""
    failure = lead_times.get("failure_epoch")
    print("\nLead-time results:")
    print(f"  failure epoch (val-accuracy collapse): {failure if failure is not None else 'none detected'}")
    header = f"  {'signal':<8}  {'fired at epoch':>14}  {'lead time (epochs)':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for signal in ("gsnr", "norm", "loss"):
        fired = lead_times.get(f"{signal}_signal")
        lead = lead_times.get(f"{signal}_lead_time")
        fired_s = str(fired) if fired is not None else "never"
        lead_s = str(lead) if lead is not None else "n/a"
        print(f"  {signal:<8}  {fired_s:>14}  {lead_s:>18}")


def main() -> None:
    """Build the config from CLI args and launch the experiment."""
    args = parse_args()
    label = EXPERIMENTS[args.exp]["label"]
    print(f"=== Experiment {args.exp}: {label} — {args.arch}, {args.epochs} epochs ===")

    config = build_config(args)
    metrics = run_baseline(config)

    print_lead_time_table(metrics["lead_times"])
    final = metrics["final_val_acc"]
    best = metrics["best_val_acc"]
    print(
        f"\n[{config.run_name}] finished in {metrics['elapsed_seconds']:.0f}s — "
        f"final val acc {final:.4f}, best val acc {best:.4f}"
    )
    print(f"[{config.run_name}] GSNR history: {metrics['csv_path']}")


if __name__ == "__main__":
    main()
