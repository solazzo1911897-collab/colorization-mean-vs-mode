"""The colouriser.

One trunk, two possible heads. Everything except the last convolution is shared
between the two losses, which is what makes the comparison in the report a
comparison of losses rather than of architectures.
"""

import torch.nn as nn

# dilation instead of striding: at 32x32 there is not much room to downsample,
# and this keeps the output at full resolution without any upsampling
DILATIONS = (1, 1, 2, 4, 8, 1)


def _block(cin, cout, dilation):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Colouriser(nn.Module):
    def __init__(self, out_channels, width=64):
        super().__init__()
        chans = [1] + [width] * len(DILATIONS)
        self.trunk = nn.Sequential(*[
            _block(chans[i], chans[i + 1], d) for i, d in enumerate(DILATIONS)
        ])
        self.head = nn.Conv2d(width, out_channels, 1)
        # start from the achromatic prediction rather than from noise
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.trunk(x))

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
