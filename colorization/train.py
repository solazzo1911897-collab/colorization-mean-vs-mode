"""Training loop shared by both losses.

Everything that differs between the two runs lives in the config file, so the
two are guaranteed to see the same trunk, the same optimiser, the same number
of steps and the same seed.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .color import ChromaBins
from .data import load_split, normalise_lightness
from .metrics import evaluate
from .model import Colouriser


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_dir(cfg, root="results"):
    payload = json.dumps(cfg, sort_keys=True).encode()
    tag = hashlib.sha1(payload).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return Path(root) / f"{stamp}_{cfg['loss']}_s{cfg['seed']}_{tag}"


def fit_bins(ab, cfg, n_images=8000):
    flat = ab[:n_images].permute(0, 2, 3, 1).reshape(-1, 2)
    return ChromaBins.fit(flat, size=cfg["bin_size"], min_count=cfg["bin_min_count"])


def train(cfg, out_root="results"):
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])

    # the training split stays in half precision on the host and is cast one
    # batch at a time; the test split is small enough not to matter
    L_tr, ab_tr, _ = load_split(cfg["data_root"], train=True, dtype=torch.float16)
    L_te, ab_te, _ = load_split(cfg["data_root"], train=False)
    x_tr = normalise_lightness(L_tr)
    x_te = normalise_lightness(L_te)

    bins = fit_bins(ab_tr.float(), cfg) if cfg["loss"] == "classification" else None
    out_channels = bins.n if bins else 2
    model = Colouriser(out_channels, width=cfg["width"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = (len(x_tr) + cfg["batch_size"] - 1) // cfg["batch_size"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch, pct_start=0.15)

    out = run_dir(cfg, out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    log = (out / "metrics.jsonl").open("w")

    print(f"{out.name}: {out_channels} output channels, "
          f"{model.n_parameters():,} parameters, on {device}", flush=True)

    started = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(len(x_tr))
        running = 0.0
        for step in range(steps_per_epoch):
            idx = order[step * cfg["batch_size"]:(step + 1) * cfg["batch_size"]]
            x = x_tr[idx].to(device, non_blocking=True).float()
            y = ab_tr[idx].to(device, non_blocking=True).float()

            pred = model(x)
            if bins is None:
                loss = F.mse_loss(pred, y)
            else:
                loss = F.cross_entropy(pred, bins.encode(y))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        stats = _validate(model, bins, x_te, ab_te, cfg)
        stats.update(epoch=epoch, train_loss=running / steps_per_epoch,
                     lr=sched.get_last_lr()[0], seconds=round(time.time() - started, 1))
        log.write(json.dumps(stats) + "\n")
        log.flush()
        print(f"  epoch {epoch + 1:2d}/{cfg['epochs']}  loss {stats['train_loss']:.4f}"
              f"  ab error {stats['ab_error']:.2f}  saturation {stats['saturation_ratio']:.1%}", flush=True)

    log.close()
    payload = {"model": model.state_dict(), "config": cfg}
    if bins is not None:
        payload["bins"] = bins.state_dict()
    torch.save(payload, out / "checkpoint.pt")
    return out


@torch.no_grad()
def _validate(model, bins, x, ab, cfg):
    """Test metrics, in chunks small enough that my laptop does not start paging.

    Decoding a classification output allocates one float per pixel per bin, so
    a chunk of a thousand images and a hundred-odd bins is a half gigabyte
    tensor before the softmax has a copy of its own. That was enough to push my
    machine into swap and take an epoch from eighty seconds to forty minutes,
    which is how I found it.
    """
    model.eval()
    device = next(model.parameters()).device
    chunk = 128 if bins is not None else 512
    preds = []
    for i in range(0, len(x), chunk):
        out = model(x[i:i + chunk].to(device))
        if bins is not None:
            out = bins.decode(out, cfg["eval_temperature"])
        preds.append(out.cpu())
    if device.type == "mps":
        torch.mps.empty_cache()
    return evaluate(torch.cat(preds), ab)
