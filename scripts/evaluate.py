"""Measure every trained run on the test set and write eval.json next to it.

For a classification run this also sweeps the decoding temperature, which is
what produces the accuracy against vividness curve in the report.

    python scripts/evaluate.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorization.data import load_split, normalise_lightness  # noqa: E402
from colorization.metrics import evaluate, per_class, saturation  # noqa: E402
from colorization.runs import list_runs, load_run, predict  # noqa: E402

# the meaningful range runs from the mode of the predicted distribution up to
# its mean at T = 1. Above 1 the softmax flattens toward the uniform, and the
# readout tends to the unweighted centroid of the bin grid, which is a constant
# with nothing to do with the image.
TEMPERATURES = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.38, 0.5, 0.65, 0.8, 1.0]


def constant_baselines(ab_train, ab_test):
    """The two predictions that involve no learning at all."""
    grey = torch.zeros_like(ab_test)
    mean = ab_train.mean(dim=(0, 2, 3)).view(1, 2, 1, 1).expand_as(ab_test)
    return {
        "grey": evaluate(grey, ab_test),
        "training_mean": evaluate(mean, ab_test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    L_tr, ab_tr, _ = load_split(args.data, train=True)
    L_te, ab_te, y_te = load_split(args.data, train=False)
    x_te = normalise_lightness(L_te)

    base = constant_baselines(ab_tr, ab_te)
    Path(args.results).mkdir(exist_ok=True)
    (Path(args.results) / "baselines.json").write_text(json.dumps(base, indent=2))
    print("baselines")
    for name, m in base.items():
        print(f"  {name:14s} ab error {m['ab_error']:6.2f}   saturation {m['saturation_ratio']:6.1%}")

    for run in list_runs(args.results):
        model, bins, cfg = load_run(run)
        out = {"config": cfg, "n_test": len(x_te)}

        if bins is None:
            pred = predict(model, None, x_te)
            out["test"] = evaluate(pred, ab_te)
            out["per_class"] = per_class(pred, ab_te, y_te)
        else:
            sweep = []
            for t in TEMPERATURES:
                pred = predict(model, bins, x_te, temperature=t)
                sweep.append({"temperature": t, **evaluate(pred, ab_te)})
            out["temperature_sweep"] = sweep
            out["n_bins"] = bins.n
            pred = predict(model, bins, x_te, temperature=cfg["eval_temperature"])
            out["test"] = evaluate(pred, ab_te)
            out["per_class"] = per_class(pred, ab_te, y_te)

        (run / "eval.json").write_text(json.dumps(out, indent=2))
        m = out["test"]
        print(f"{run.name}: ab error {m['ab_error']:6.2f}   saturation {m['saturation_ratio']:6.1%}"
              f"   within 15 {m['within_15']:6.1%}")


if __name__ == "__main__":
    main()
