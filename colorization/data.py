"""CIFAR-10, converted to Lab once and cached.

The colouriser only ever sees the L channel. The class labels are loaded too,
but they are used exclusively in the per-class analysis of Section 4, never
during training.
"""

from pathlib import Path

import torch
from torchvision import datasets

from .color import rgb_to_lab

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def _build_cache(root: Path, train: bool):
    ds = datasets.CIFAR10(root=str(root), train=train, download=True)
    rgb = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(ds.targets, dtype=torch.long)

    # in chunks, because converting all fifty thousand at once asks for more
    # memory than my laptop is willing to hand over
    lab = torch.empty_like(rgb)
    for i in range(0, len(rgb), 4096):
        lab[i:i + 4096] = rgb_to_lab(rgb[i:i + 4096])

    return {"lab": lab.half(), "labels": labels}


def load_split(root="data", train=True, dtype=torch.float32):
    """Lightness, chroma and labels for one split.

    Holding the training split in float32 costs 614 MB, which on my eight
    gigabyte laptop is the difference between training and swapping, so callers
    that only need it batch by batch ask for float16 and cast as they go. Half
    precision resolves to about 0.06 Lab units at these magnitudes, four orders
    of magnitude below anything I measure here.
    """
    root = Path(root)
    cache = root / f"cifar10_lab_{'train' if train else 'test'}.pt"
    if not cache.exists():
        root.mkdir(parents=True, exist_ok=True)
        torch.save(_build_cache(root, train), cache)
    d = torch.load(cache, weights_only=True)
    lab = d["lab"].to(dtype)
    return lab[:, :1], lab[:, 1:], d["labels"]


def normalise_lightness(L):
    """L lives in [0, 100]; the network prefers something centred on zero."""
    return L / 50.0 - 1.0
