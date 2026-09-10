"""The per-pixel version of the per-class prediction of Section 4.

Grouping by class asks whether a whole category is ambiguous. This asks the
same question one pixel at a time, using the classification model's own
predictive entropy as the measure of ambiguity, and testing whether vividness
falls where the entropy is high.

    python scripts/entropy.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorization.data import load_split, normalise_lightness  # noqa: E402
from colorization.runs import list_runs, load_run, predict, read_config  # noqa: E402

BLUE, ORANGE = "#1f6fb4", "#d95f0e"
N_BUCKETS = 10


@torch.no_grad()
def predictive_entropy(model, x, chunk=128):
    """Entropy of the predicted bin distribution, one number per pixel."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x), chunk):
        logp = torch.log_softmax(model(x[i:i + chunk].to(device)), dim=1)
        out.append((-(logp.exp() * logp).sum(1)).cpu())
    return torch.cat(out)


def chroma(ab):
    return ab.pow(2).sum(1).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    runs = {"l2": [], "classification": []}
    for r in list_runs(args.results):
        runs[read_config(r)["loss"]].append(r)

    L_te, ab_te, _ = load_split(args.data, train=False)
    x = normalise_lightness(L_te[:args.n])
    truth = ab_te[:args.n]

    m_l2, _, _ = load_run(runs["l2"][0])
    m_cls, bins, _ = load_run(runs["classification"][0])

    h = predictive_entropy(m_cls, x).flatten()
    c_true = chroma(truth).flatten()
    c_l2 = chroma(predict(m_l2, None, x)).flatten()
    c_cls = chroma(predict(m_cls, bins, x, temperature=1.0)).flatten()

    # equal-count buckets, so every point carries the same number of pixels
    edges = torch.quantile(h[::7].float(), torch.linspace(0, 1, N_BUCKETS + 1))
    edges[0], edges[-1] = -float("inf"), float("inf")
    idx = torch.bucketize(h, edges[1:-1])

    rows = []
    for b in range(N_BUCKETS):
        m = idx == b
        rows.append({
            "bucket": b,
            "entropy": h[m].mean().item(),
            "n_pixels": int(m.sum()),
            "chroma_true": c_true[m].mean().item(),
            "ratio_l2": (c_l2[m].mean() / c_true[m].mean()).item(),
            "ratio_cls": (c_cls[m].mean() / c_true[m].mean()).item(),
        })

    e = np.array([r["entropy"] for r in rows])
    corr = {k: float(np.corrcoef(e, [r[k] for r in rows])[0, 1]) for k in ("ratio_l2", "ratio_cls")}
    out = {"buckets": rows, "correlation": corr, "n_images": args.n}
    (Path(args.results) / "entropy.json").write_text(json.dumps(out, indent=2))

    for r in rows:
        print(f"  entropy {r['entropy']:5.2f}  true chroma {r['chroma_true']:6.2f}  "
              f"vividness  L2 {100*r['ratio_l2']:5.1f}%   classification {100*r['ratio_cls']:5.1f}%")
    print(f"correlation with entropy: squared error {corr['ratio_l2']:+.2f}, "
          f"classification {corr['ratio_cls']:+.2f}")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200,
                         "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(3.2, 2.1), layout="constrained")
    ax.plot(e, [100 * r["ratio_l2"] for r in rows], "-s", color=ORANGE, ms=3.5, lw=1,
            label="squared error")
    ax.plot(e, [100 * r["ratio_cls"] for r in rows], "-o", color=BLUE, ms=3.5, lw=1,
            label="classification, T = 1")
    ax.axhline(100, color="k", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel("predictive entropy of the pixel, nats")
    ax.set_ylabel("vividness, % of ground truth")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.savefig(Path(args.out) / "fig6_entropy.png")
    print("figure written to", Path(args.out) / "fig6_entropy.png")


if __name__ == "__main__":
    main()
