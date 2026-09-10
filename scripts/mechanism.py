"""Check that the two models learned the same thing and differ only in readout.

If the squared error really does drive a model to the mean of the conditional
colour distribution, then the L2 network's output should coincide with the
expectation of the classification network's predicted distribution, taken at
temperature 1. Nothing forces this: the two were trained separately, with
different losses and different output layers.

Saying that the distance between them is small needs a scale, and the distance
to the ground truth is the wrong one: it says the two are closer to each other
than either is to the answer, not that they agree. The right scale is how far
apart two runs of the *same* loss are when they differ only in the seed, which
is the floor that any pair of trained networks sits above. That is computed
here too.

    python scripts/mechanism.py
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorization.data import load_split, normalise_lightness  # noqa: E402
from colorization.runs import list_runs, load_run, predict, read_config  # noqa: E402

MODE_TEMPERATURE = 0.02


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

    # every prediction once, so that the cross-loss and the within-loss
    # comparisons below are computed from exactly the same tensors
    pred = {}
    for r in runs["l2"]:
        model, _, _ = load_run(r)
        pred[("l2", read_config(r)["seed"])] = predict(model, None, x)
    for r in runs["classification"]:
        model, bins, _ = load_run(r)
        seed = read_config(r)["seed"]
        pred[("expectation", seed)] = predict(model, bins, x, temperature=1.0)
        pred[("mode", seed)] = predict(model, bins, x, temperature=MODE_TEMPERATURE)

    seeds = sorted({s for _, s in pred})

    rows = []
    for seed_l2 in seeds:
        for seed_cls in seeds:
            rows.append({
                "seed_l2": seed_l2,
                "seed_cls": seed_cls,
                "same_seed": seed_l2 == seed_cls,
                "l2_vs_expectation": agreement(pred[("l2", seed_l2)], pred[("expectation", seed_cls)]),
                "l2_vs_mode": agreement(pred[("l2", seed_l2)], pred[("mode", seed_cls)]),
                "l2_vs_truth": agreement(pred[("l2", seed_l2)], truth),
            })
            r = rows[-1]
            print(f"L2 seed {seed_l2} against classification seed {seed_cls}: "
                  f"distance to its expectation {r['l2_vs_expectation']['mean_distance']:.2f}, "
                  f"to its mode {r['l2_vs_mode']['mean_distance']:.2f}, "
                  f"to the truth {r['l2_vs_truth']['mean_distance']:.2f}")

    # the scale: two runs of one loss, differing only in the seed
    within = {}
    for group in ("l2", "expectation", "mode"):
        within[group] = [
            {"seeds": [a, b], **agreement(pred[(group, a)], pred[(group, b)])}
            for a, b in itertools.combinations(seeds, 2)
        ]
        ds = [p["mean_distance"] for p in within[group]]
        print(f"within {group:12s}: {sum(ds) / len(ds):.2f} "
              f"[{min(ds):.2f}, {max(ds):.2f}] over {len(ds)} seed pairs")

    out = Path(args.results) / "mechanism.json"
    out.write_text(json.dumps({"pairs": rows, "within": within}, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
