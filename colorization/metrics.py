"""The two quantities the report trades off against each other, plus the
per-class ambiguity used to explain the trade.

Both are computed in the ab plane and ignore lightness, which the model is
given and never has to guess.
"""

import torch


def ab_error(pred, target):
    """Mean euclidean distance in the ab plane, one number per batch of images."""
    return (pred - target).pow(2).sum(1).sqrt().mean().item()


def within(pred, target, threshold=15.0):
    """Fraction of pixels whose colour lands within threshold of the true one."""
    d = (pred - target).pow(2).sum(1).sqrt()
    return (d < threshold).float().mean().item()


def saturation(ab):
    """Mean chroma. Grey is zero, and the ground truth sets the scale."""
    return ab.pow(2).sum(1).sqrt().mean().item()


def spread(ab):
    """Mean distance from the average predicted colour.

    Saturation on its own can be fooled: a model that paints every pixel the
    same warm beige scores a respectable chroma while producing exactly one
    colour. This is zero for any constant prediction, so the two together say
    both how far from grey the output is and how much it varies.
    """
    mean = ab.mean(dim=(0, 2, 3)).view(1, 2, 1, 1)
    return (ab - mean).pow(2).sum(1).sqrt().mean().item()


def evaluate(pred, target):
    sat_true, sat_pred = saturation(target), saturation(pred)
    spread_true, spread_pred = spread(target), spread(pred)
    return {
        "ab_error": ab_error(pred, target),
        "within_15": within(pred, target),
        "saturation": sat_pred,
        "saturation_true": sat_true,
        "saturation_ratio": sat_pred / sat_true,
        "spread": spread_pred,
        "spread_true": spread_true,
        "spread_ratio": spread_pred / spread_true,
    }


def class_chroma_variance(ab, labels, n_classes=10):
    """How much the colour varies inside each class.

    This is the empirical variance of the ab vector around the class mean,
    taken over every pixel of every image with that label. A class whose colour
    is dictated by what it depicts (frogs, ships) scores low; one whose colour
    is a free choice of the manufacturer (cars, trucks) scores high.
    """
    out = []
    for c in range(n_classes):
        chunk = ab[labels == c]
        flat = chunk.permute(0, 2, 3, 1).reshape(-1, 2)
        out.append((flat - flat.mean(0)).pow(2).sum(1).mean().item())
    return torch.tensor(out)


def class_instance_chroma_variance(ab, labels, n_classes=10):
    """How much the average colour of a whole image varies between instances.

    class_chroma_variance counts every drop of chroma variation inside a class,
    including the variation within a single photograph: the grass beside the
    deer is not ambiguity, it is structure the network can read straight off
    the lightness. What is genuinely ambiguous is whether *this* truck is red
    or white, and that is the spread of per-image mean colour across the class.

    Both measures are reported. This is the one the argument is about.
    """
    per_image = ab.mean(dim=(2, 3))
    out = []
    for c in range(n_classes):
        v = per_image[labels == c]
        out.append((v - v.mean(0)).pow(2).sum(1).mean().item())
    return torch.tensor(out)


def per_class(pred, target, labels, n_classes=10):
    rows = []
    for c in range(n_classes):
        m = labels == c
        rows.append({
            "class": int(c),
            "ab_error": ab_error(pred[m], target[m]),
            "saturation": saturation(pred[m]),
            "saturation_true": saturation(target[m]),
            "saturation_ratio": saturation(pred[m]) / saturation(target[m]),
        })
    return rows
