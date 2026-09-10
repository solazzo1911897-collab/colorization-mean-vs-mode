"""Loading back what a training run left on disk."""

import json
from pathlib import Path

import torch
import yaml

from .color import ChromaBins
from .model import Colouriser
from .train import pick_device


def list_runs(root="results"):
    return sorted(p for p in Path(root).iterdir() if (p / "checkpoint.pt").exists())


def load_run(path, device=None):
    path = Path(path)
    ckpt = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = device or pick_device(cfg["device"])

    bins = ChromaBins.from_state_dict(ckpt["bins"]) if "bins" in ckpt else None
    model = Colouriser(bins.n if bins else 2, width=cfg["width"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), bins, cfg


def history(path):
    lines = (Path(path) / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def read_config(path):
    return yaml.safe_load((Path(path) / "config.yaml").read_text())


@torch.no_grad()
def predict(model, bins, x, temperature=None, chunk=None):
    """ab prediction for a batch of normalised lightness images."""
    device = next(model.parameters()).device
    chunk = chunk or (128 if bins is not None else 512)
    out = []
    for i in range(0, len(x), chunk):
        raw = model(x[i:i + chunk].to(device))
        out.append(raw.cpu() if bins is None else bins.decode(raw, temperature).cpu())
    return torch.cat(out)


@torch.no_grad()
def predict_logits(model, x, chunk=128):
    device = next(model.parameters()).device
    return torch.cat([model(x[i:i + chunk].to(device)).cpu() for i in range(0, len(x), chunk)])
