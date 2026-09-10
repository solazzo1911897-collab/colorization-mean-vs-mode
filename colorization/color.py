"""Conversions between sRGB and CIE Lab, plus the quantisation of the ab plane.

Lab is used throughout because euclidean distance in it is a reasonable stand-in
for perceived colour difference, which matters both for the bins and for the
error metric.
"""

import torch

_RGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = torch.inverse(_RGB_TO_XYZ)

# D65, the white point sRGB is defined against
_WHITE = torch.tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)

_DELTA = 6.0 / 29.0


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def _f(t):
    return torch.where(t > _DELTA ** 3, t.clamp(min=1e-8) ** (1 / 3), t / (3 * _DELTA ** 2) + 4 / 29)


def _f_inv(t):
    return torch.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4 / 29))


def rgb_to_lab(rgb):
    """rgb in [0, 1] with shape (B, 3, H, W) -> Lab with L in [0, 100]."""
    lin = _srgb_to_linear(rgb)
    xyz = torch.einsum("ij,bjhw->bihw", _RGB_TO_XYZ.to(rgb), lin)
    fx, fy, fz = _f(xyz / _WHITE.to(rgb)).unbind(1)
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


def lab_to_rgb(lab):
    L, a, b = lab.unbind(1)
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = _f_inv(torch.stack([fx, fy, fz], dim=1)) * _WHITE.to(lab)
    lin = torch.einsum("ij,bjhw->bihw", _XYZ_TO_RGB.to(lab), xyz)
    return _linear_to_srgb(lin).clamp(0, 1)


class ChromaBins:
    """The ab plane cut into a square grid, keeping only the cells the data uses.

    Most of the ab plane is out of gamut for sRGB, and of what is left CIFAR-10
    only visits a part, so quantising the full plane would spend most of the
    output layer on colours that never occur.
    """

    # the ab plane never leaves this box for colours sRGB can actually display
    LIMIT = 120.0
    # the lookup grid is finer than the bins so that points falling in a dropped
    # cell are still assigned to the centre nearest to them, not to their cell
    LUT_REFINE = 4

    def __init__(self, centers, size):
        self.centers = centers          # (Q, 2)
        self.size = size
        self._lut = self._build_lut()

    @property
    def n(self):
        return len(self.centers)

    def _build_lut(self):
        """Nearest kept centre for every point of a grid finer than the bins.

        Doing this once turns encoding a batch from a distance matrix against
        every bin into an integer lookup. Measured by scripts/bins_check.py on
        two million real pixels it picks the same centre as the exact
        calculation 97.2% of the time, and the ones it misses land 0.003%
        further away on average, which is not worth a matrix multiplication per
        batch.
        """
        step = self.size / self.LUT_REFINE
        side = int(2 * self.LIMIT / step)
        coords = (torch.arange(side).float() + 0.5) * step - self.LIMIT
        cells = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(-1, 2)
        return torch.cdist(cells, self.centers).argmin(1).view(side, side)

    @classmethod
    def fit(cls, ab, size=10.0, min_count=20):
        """ab has shape (N, 2). Cells with fewer than min_count samples are dropped."""
        idx = torch.floor(ab / size).long()
        keys, counts = torch.unique(idx, dim=0, return_counts=True)
        keys = keys[counts >= min_count]
        centers = (keys.float() + 0.5) * size
        # sorting keeps the bin order stable across runs, so checkpoints stay comparable
        order = torch.argsort(centers[:, 0] * 1e4 + centers[:, 1])
        return cls(centers[order].contiguous(), size)

    def encode(self, ab):
        """(B, 2, H, W) -> (B, H, W) long, the index of the nearest kept centre."""
        if self._lut.device != ab.device:
            self._lut = self._lut.to(ab.device)
        side = self._lut.shape[0]
        cell = ((ab + self.LIMIT) * self.LUT_REFINE / self.size).long().clamp(0, side - 1)
        return self._lut[cell[:, 0], cell[:, 1]]

    def decode(self, logits, temperature):
        """Annealed expectation of the predicted distribution over bins.

        temperature -> 0 gives the most likely bin, temperature = 1 gives the
        mean of the distribution. Everything interesting happens in between.
        """
        b, q, h, w = logits.shape
        p = torch.softmax(logits.permute(0, 2, 3, 1).reshape(-1, q) / temperature, dim=1)
        ab = p @ self.centers.to(logits)
        return ab.view(b, h, w, 2).permute(0, 3, 1, 2)

    def state_dict(self):
        return {"centers": self.centers, "size": self.size}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["centers"], d["size"])
