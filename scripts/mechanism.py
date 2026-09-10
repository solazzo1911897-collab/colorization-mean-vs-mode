"""Check that the two models learned the same thing and differ only in readout.

If the squared error really does drive a model to the mean of the conditional
colour distribution, then the L2 network's output should coincide with the
expectation of the classification network's predicted distribution, taken at
temperature 1. Nothing forces this: the two were trained separately, with
different losses and different output layers.

    python scripts/mechanism.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorization.data import load_split, normalise_lightness  # noqa: E402
from colorization.runs import list_runs, load_run, predict, read_config  # noqa: E402


def agreement(u, v):
    """Distance and correlation between two ab predictions of the same images."""
    d = (u - v).pow(2).sum(1).sqrt()
    a = torch.stack([u[:, 0].flatten(), v[:, 0].flatten()])
    b = torch.stack([u[:, 1].flatten(), v[:, 1].flatten()])
    return {
        "mean_distance": d.mean().item(),
        "median_distance": d.median().item(),
        "corr_a": torch.corrcoef(a)[0, 1].item(),
        "corr_b": torch.corrcoef(b)[0, 1].item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    L_te, ab_te, _ = load_split(args.data, train=False)
    x = normalise_lightness(L_te[:args.n])
    truth = ab_te[:args.n]

    runs = {"l2": [], "classification": []}
    for r in list_runs(args.results):
        runs[read_config(r)["loss"]].append(r)

    rows = []
    for r_l2 in runs["l2"]:
        seed_l2 = read_config(r_l2)["seed"]
        m_l2, _, _ = load_run(r_l2)
        pred_l2 = predict(m_l2, None, x)

        for r_cls in runs["classification"]:
            seed_cls = read_config(r_cls)["seed"]
            m_cls, bins, _ = load_run(r_cls)
            expectation = predict(m_cls, bins, x, temperature=1.0)
            mode = predict(m_cls, bins, x, temperature=0.02)

            rows.append({
                "seed_l2": seed_l2,
                "seed_cls": seed_cls,
                "same_seed": seed_l2 == seed_cls,
                "l2_vs_expectation": agreement(pred_l2, expectation),
                "l2_vs_mode": agreement(pred_l2, mode),
                "l2_vs_truth": agreement(pred_l2, truth),
            })
            r = rows[-1]
            print(f"L2 seed {seed_l2} against classification seed {seed_cls}: "
                  f"distance to its expectation {r['l2_vs_expectation']['mean_distance']:.2f}, "
                  f"to its mode {r['l2_vs_mode']['mean_distance']:.2f}, "
                  f"to the truth {r['l2_vs_truth']['mean_distance']:.2f}")

    out = Path(args.results) / "mechanism.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
