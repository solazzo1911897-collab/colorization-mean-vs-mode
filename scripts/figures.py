"""Rebuild every figure in the report from what is in results/.

Nothing here trains anything, so it is cheap to run again after changing a
label or a colour.

    python scripts/figures.py
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
from colorization.color import lab_to_rgb  # noqa: E402
from colorization.data import CLASS_NAMES, load_split, normalise_lightness  # noqa: E402
from colorization.metrics import (class_chroma_variance,  # noqa: E402
                                  class_instance_chroma_variance)
from colorization.runs import list_runs, load_run, predict, read_config  # noqa: E402

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

BLUE, ORANGE, GREY = "#1f6fb4", "#d95f0e", "#8a8a8a"


def to_image(L, ab):
    return lab_to_rgb(torch.cat([L, ab], dim=1)).permute(0, 2, 3, 1).numpy()


def group_runs(results):
    runs = {"l2": [], "classification": []}
    for r in list_runs(results):
        runs[read_config(r)["loss"]].append(r)
    return runs


def fig_qualitative(runs, L, ab, out, picks, temperature):
    m_l2, _, _ = load_run(runs["l2"][0])
    m_cls, bins, cfg = load_run(runs["classification"][0])
    x = normalise_lightness(L)

    rows = [
        ("input", torch.zeros_like(ab)),
        ("squared error", predict(m_l2, None, x)),
        (f"classification, T = {temperature}", predict(m_cls, bins, x, temperature=temperature)),
        ("ground truth", ab),
    ]

    fig, axes = plt.subplots(len(rows), len(picks), figsize=(0.62 * len(picks), 0.62 * len(rows) + 0.2))
    for r, (name, chroma) in enumerate(rows):
        imgs = to_image(L, chroma)
        for c, k in enumerate(picks):
            axes[r, c].imshow(imgs[k])
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(out / "fig1_qualitative.png")
    plt.close(fig)


def fig_frontier(runs, results, out):
    """Colour accuracy against how colourful the output is.

    Two panels because one measure is not enough: chroma alone would give a
    respectable score to a model that painted every pixel the same beige, and
    the spread panel is what shows it did not.
    """
    baselines = json.loads((Path(results) / "baselines.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharex=True, layout="constrained")

    for ax, key, ylabel in [(axes[0], "saturation_ratio", "vividness, % of ground truth"),
                            (axes[1], "spread_ratio", "colour variety, % of ground truth")]:
        for i, r in enumerate(runs["classification"]):
            sweep = json.loads((r / "eval.json").read_text())["temperature_sweep"]
            ax.plot([s["ab_error"] for s in sweep], [100 * s[key] for s in sweep],
                    "-o", color=BLUE, ms=2.5, lw=1, alpha=0.85,
                    label="classification, T from 0.02 to 1" if i == 0 else None)
            if i == 0:
                for s in sweep:
                    if s["temperature"] in (0.02, 0.38, 1.0):
                        ax.annotate(f"T={s['temperature']:g}", (s["ab_error"], 100 * s[key]),
                                    textcoords="offset points", xytext=(4, 3),
                                    fontsize=6, color=BLUE)

        for i, r in enumerate(runs["l2"]):
            m = json.loads((r / "eval.json").read_text())["test"]
            ax.plot(m["ab_error"], 100 * m[key], "s", color=ORANGE, ms=5,
                    label="squared error" if i == 0 else None)

        for name, style in [("grey", "^"), ("training_mean", "v")]:
            m = baselines[name]
            ax.plot(m["ab_error"], 100 * m[key], style, color=GREY, ms=4.5,
                    label=name.replace("_", " ") if ax is axes[0] else None)

        ax.axhline(100, color="k", lw=0.7, ls=":", zorder=0)
        ax.set_xlabel("mean colour error in the ab plane")
        ax.set_ylabel(ylabel)

    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.savefig(out / "fig2_frontier.png")
    plt.close(fig)


def fig_per_class(runs, ab_train, y_train, out):
    """Vividness against two different readings of how ambiguous a class is.

    The left panel is the one the argument is about: how much the overall
    colour of an image varies between instances of the class. The right panel
    counts every pixel, which mixes that together with variation inside a
    single photograph, and is shown because it is the obvious measure and it
    does not work.
    """
    axes_spec = [
        (class_instance_chroma_variance(ab_train, y_train).sqrt(),
         "spread of image colour between instances"),
        (class_chroma_variance(ab_train, y_train).sqrt(),
         "spread over all pixels of the class"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True, layout="constrained")

    corrs = {}
    for ax, (var, xlabel) in zip(axes, axes_spec):
        for label, group, colour in [("squared error", "l2", ORANGE),
                                     ("classification", "classification", BLUE)]:
            ratios = np.array([[c["saturation_ratio"] for c in
                                json.loads((r / "eval.json").read_text())["per_class"]]
                               for r in runs[group]])
            mean = 100 * ratios.mean(0)
            ax.scatter(var, mean, s=16, color=colour, zorder=3,
                       label=label if ax is axes[0] else None)
            if group == "l2":
                for i, name in enumerate(CLASS_NAMES):
                    ax.annotate(name, (var[i], mean[i]), fontsize=5,
                                textcoords="offset points", xytext=(4, -1.5),
                                color="#444444")
                r = float(np.corrcoef(var.numpy(), mean)[0, 1])
                corrs["instances" if ax is axes[0] else "pixels"] = r
                ax.set_title(f"correlation {r:+.2f}", fontsize=7)
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("vividness, % of ground truth")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left",
                   bbox_to_anchor=(0.0, -0.28), ncol=2)
    fig.savefig(out / "fig3_per_class.png")
    plt.close(fig)
    return corrs


def fig_mechanism(runs, L, ab, out, n=400):
    m_l2, _, _ = load_run(runs["l2"][0])
    m_cls, bins, _ = load_run(runs["classification"][0])
    x = normalise_lightness(L[:n])
    p_l2 = predict(m_l2, None, x)
    p_mean = predict(m_cls, bins, x, temperature=1.0)
    p_mode = predict(m_cls, bins, x, temperature=0.02)

    edge = float(np.percentile(np.abs(np.concatenate(
        [p_l2[:, 0].flatten().numpy(), p_mean[:, 0].flatten().numpy(),
         p_mode[:, 0].flatten().numpy()])), 99.5))
    lim = [-edge, edge]

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.1), sharex=True, sharey=True, layout="constrained")
    for ax, other, title in [(axes[0], p_mean, "expectation of the distribution, T = 1"),
                             (axes[1], p_mode, "most likely bin, T = 0.02")]:
        u, v = p_l2[:, 0].flatten().numpy(), other[:, 0].flatten().numpy()
        ax.hexbin(u, v, gridsize=60, bins="log", cmap="Blues", mincnt=1, linewidths=0,
                  extent=(*lim, *lim))
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("squared error model, a channel")
    axes[0].set_ylabel("classification model, a channel")
    fig.savefig(out / "fig4_mechanism.png")
    plt.close(fig)


def fig_temperature_strip(runs, L, ab, out, picks, temps=(0.02, 0.1, 0.2, 0.38, 0.65, 1.0)):
    m_cls, bins, _ = load_run(runs["classification"][0])
    x = normalise_lightness(L)
    rows = [(f"T = {t:g}", predict(m_cls, bins, x, temperature=t)) for t in temps]
    rows.append(("ground truth", ab))

    fig, axes = plt.subplots(len(rows), len(picks), figsize=(0.62 * len(picks), 0.62 * len(rows)))
    for r, (name, chroma) in enumerate(rows):
        imgs = to_image(L, chroma)
        for c, k in enumerate(picks):
            axes[r, c].imshow(imgs[k])
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(out / "fig5_temperature.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--temperature", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    runs = group_runs(args.results)

    L_te, ab_te, _ = load_split(args.data, train=False)
    _, ab_tr, y_tr = load_split(args.data, train=True)

    # the qualitative grid is a single column in the report, so it stays narrow;
    # the temperature strip is an appendix figure and can be wider
    picks = [7, 12, 19, 25, 33, 41, 47, 52, 61, 70]
    fig_qualitative(runs, L_te[:200], ab_te[:200], out, picks, args.temperature)
    fig_frontier(runs, args.results, out)
    corrs = fig_per_class(runs, ab_tr, y_tr, out)
    fig_mechanism(runs, L_te, ab_te, out)
    (Path(args.results) / "per_class.json").write_text(json.dumps(corrs, indent=2))
    fig_temperature_strip(runs, L_te[:200], ab_te[:200], out,
                          [7, 12, 19, 25, 33, 47, 52, 61, 70, 88])
    print("figures written to", out)


if __name__ == "__main__":
    main()
