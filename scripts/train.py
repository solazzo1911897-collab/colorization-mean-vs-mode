"""Train one colouriser.

    python scripts/train.py config/l2.yaml --seed 1
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorization.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--data")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.data is not None:
        cfg["data_root"] = args.data

    train(cfg, out_root=args.results)


if __name__ == "__main__":
    main()
