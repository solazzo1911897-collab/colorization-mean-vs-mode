"""What the quantisation of the ab plane costs, measured rather than asserted.

Two things are checked here, both of which the report and the README lean on.

The lookup table. Encoding a pixel to its bin goes through an integer lookup on
a grid finer than the bins rather than a distance matrix against every centre.
That is an approximation, and this measures how often it disagrees with the
exact nearest centre and how much further away it lands when it does.

The ceiling. Dropping the cells the training set barely visits is the obvious
thing to suspect when a colouriser comes out desaturated: if the grid itself
cannot represent vivid colour, no readout can recover it. Quantising the ground
truth to the kept centres and measuring it as if it were a prediction settles
that, because it is the best any model over these bins could possibly do.

    python scripts/bins_check.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorization.data import load_split  # noqa: E402
from colorization.metrics import evaluate  # noqa: E402
from colorization.train import fit_bins  # noqa: E402

N_PIXELS = 2_000_000
CHUNK = 100_000


def lookup_table_error(bins, pixels):
    """How the integer lookup compares with the exact nearest centre."""
    agree, excess, total = 0, 0.0, 0.0
    for i in range(0, len(pixels), CHUNK):
        p = pixels[i:i + CHUNK]
        approx = bins.encode(p.T.reshape(1, 2, -1, 1))[0, :, 0]
        d = torch.cdist(p, bins.centers)
        exact = d.argmin(1)
        agree += int((approx == exact).sum())
        d_approx = d.gather(1, approx.view(-1, 1)).sum().item()
        d_exact = d.gather(1, exact.view(-1, 1)).sum().item()
        excess += d_approx - d_exact
        total += d_exact
    return agree / len(pixels), excess / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="config/classification.yaml")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())

    _, ab_tr, _ = load_split(args.data, train=True, dtype=torch.float16)
    _, ab_te, _ = load_split(args.data, train=False)
    bins = fit_bins(ab_tr.float(), cfg)

    flat = ab_tr.float().permute(0, 2, 3, 1).reshape(-1, 2)
    generator = torch.Generator().manual_seed(0)
    sample = flat[torch.randint(0, len(flat), (N_PIXELS,), generator=generator)]

    agree, excess = lookup_table_error(bins, sample)

    # a pixel is covered when the centre it is assigned to is the one of its own
    # cell, that is when it is no further than half a cell diagonal away
    half_diagonal = cfg["bin_size"] * 0.5 * 2 ** 0.5
    encoded = bins.encode(ab_te)
    quantised = bins.centers[encoded].permute(0, 3, 1, 2)
    distance = (quantised - ab_te).pow(2).sum(1).sqrt()
    ceiling = evaluate(quantised, ab_te)

    out = {
        "n_bins": bins.n,
        "bin_size": cfg["bin_size"],
        "min_count": cfg["bin_min_count"],
        "lut_agreement": agree,
        "lut_excess_distance": excess,
        "coverage": (distance <= half_diagonal).float().mean().item(),
        "quantised_truth": ceiling,
    }
    (Path(args.results) / "bins.json").write_text(json.dumps(out, indent=2))

    print(f"{bins.n} bins of {cfg['bin_size']:g} Lab units")
    print(f"lookup table agrees with the exact nearest centre {100 * agree:.2f}% of the time,")
    print(f"  and lands {100 * excess:.4f}% further away on average when it does not")
    print(f"{100 * out['coverage']:.2f}% of test pixels fall inside a kept cell")
    print("ground truth quantised to the kept centres:")
    print(f"  colour error {ceiling['ab_error']:.2f}, "
          f"vividness {100 * ceiling['saturation_ratio']:.1f}% of the truth")
    print(f"written to {Path(args.results) / 'bins.json'}")


if __name__ == "__main__":
    main()
