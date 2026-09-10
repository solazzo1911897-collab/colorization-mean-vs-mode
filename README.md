# What the loss picks when the colour is ambiguous

Project for Deep Learning and Applied AI, Sapienza, a.y. 2025/26.

Colourising a black and white photograph has no single right answer. A grey car
could have been red, blue or white, and nothing in the greyscale image decides
between them. This repository asks what a network does when it is trained on a
problem like that, and shows that the answer is decided by the loss and not by
the architecture or the amount of training.

Two colourisers are trained on CIFAR-10. They share the same trunk, the same
optimiser, the same schedule, the same number of steps and the same seeds, and
differ only in the last layer and the loss:

- **squared error** on the two chroma channels of Lab,
- **cross-entropy** over 124 quantised colour bins, read back out at a
  temperature that interpolates between the mean of the predicted distribution
  and its mode.

Both are systematically desaturated, keeping around 62% of the ground truth's
chroma. Sweeping the temperature turns the second model's single prediction into
a curve trading colour accuracy against colour variety, and the first model sits
on that curve rather than off it. Read out at temperature 1, the second
reproduces the first pixel by pixel: 3.11 Lab units apart, against the 12.49 that
separates either from the truth. The scale that makes 3.11 small is not that
distance but the run-to-run floor, since two squared error models differing only
in their seed are 2.73 apart and two classification models 2.45: changing the
loss moves the output about as far as changing the initialisation does. That is
what identifies the desaturation as a property of the question the loss asks
rather than of any particular model or run.

A prediction that did not survive is reported too: the desaturation was expected
to be worst where colour is least predictable, and neither grouping by CIFAR-10
class nor sorting pixels by the model's own predictive entropy shows any such
relation.

## What is taken from previous work, and what is not

The idea of posing colourisation as a classification over a quantised grid of
`ab` bins, and of decoding the predicted distribution with an annealed mean
whose temperature interpolates between that distribution's mean and its mode,
is due to Zhang, Isola and Efros (ECCV 2016). Larsson, Maire and Shakhnarovich
(ECCV 2016) proposed predicting per-pixel colour histograms independently in the
same year. Neither the method nor the observation that a squared error produces
desaturated output is claimed as new here; the latter is a well established
property of the loss, stated for video prediction by Mathieu, Couprie and LeCun
(ICLR 2016).

What this repository adds is a controlled measurement of the trade. The two
losses are compared with the trunk, the optimiser, the schedule, the step count
and the seeds all held fixed, so that the only difference is the loss; the
frontier is traced by one knob rather than by two separately tuned systems; and
the readout comparison in `scripts/mechanism.py` tests whether the two networks
have in fact learned the same conditional distribution and differ only in how it
is collapsed to one colour, read against the only scale that makes the question
answerable, which is how far apart two runs of one loss are.

Three deliberate departures from Zhang et al., all of which make this a smaller
experiment rather than an improved method:

- **No class rebalancing.** Their loss reweights rare colours to counteract the
  dominance of desaturated ones. That reweighting is itself a second mechanism
  for increasing vividness, and including it would confound the very trade being
  measured here, so the cross-entropy is left plain.
- **Bins fitted to this dataset.** They use a fixed 313-bin grid covering the
  sRGB gamut; here the same 10-unit grid is cut down to the cells CIFAR-10
  actually visits.
- **CIFAR-10 at 32 by 32 rather than ImageNet.** The full design is six models
  and a thirteen-point temperature sweep, and at this scale it fits in an
  afternoon on a laptop.

The dataset is CIFAR-10 (Krizhevsky, 2009), downloaded by `torchvision` from
`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`. All code here is
written for this project; the colour space conversion in `colorization/color.py`
is implemented from the sRGB and CIE definitions rather than taken from a
library, and is checked against the published Lab values of the sRGB primaries.

## Running it

Requires [uv](https://docs.astral.sh/uv/). Apple MPS is picked up
automatically, CPU is the fallback, and `device: cuda` in a config selects a
CUDA card. The full set of runs takes about three hours on an M2.

```bash
uv sync
./scripts/run_all.sh
```

That trains six models (two losses, three seeds each), evaluates them, runs the
readout comparison and writes every figure into `figures/`. CIFAR-10 downloads
itself into `data/` on first use, about 170 MB.

The steps individually, if you want them one at a time:

```bash
uv run python scripts/train.py config/l2.yaml --seed 0
uv run python scripts/train.py config/classification.yaml --seed 0
uv run python scripts/evaluate.py     # test metrics, and the temperature sweep
uv run python scripts/mechanism.py    # squared error against the annealed mean
uv run python scripts/bins_check.py   # what the quantisation costs
uv run python scripts/figures.py      # everything in figures/
uv run python scripts/entropy.py      # vividness against per-pixel ambiguity
```

## Layout

```
config/          one file per loss, every knob in it
colorization/    the library: colour space, data, model, metrics, training loop
scripts/         the entry points listed above
results/         one directory per run, holding its config and its metrics
figures/         output only, safe to delete
```

Each run directory is named by date, loss and seed, and contains the exact
config it was run with, a `metrics.jsonl` with one line per epoch, and an
`eval.json` written by `scripts/evaluate.py`.

## What is and is not committed

`data/` is not: `torchvision` fetches it on first use, and 170 MB of somebody
else's copy of CIFAR-10 does not belong here.

The six checkpoints are, at 4.6 MB in total. They take three hours to retrain
and every figure except the frontier needs them, so committing them is the
difference between rebuilding the whole report in two minutes and rebuilding it
in an afternoon. The per-epoch metrics and the evaluation output are committed
for the same reason.

So a fresh clone reproduces every number in the report immediately, and every
figure once `torchvision` has fetched CIFAR-10, without training anything:

```bash
uv run python scripts/report_numbers.py
uv run python scripts/figures.py
```

## Notes on the setup

**Colour space.** Lab, so that lightness can be held fixed and only the chroma
guessed, and because euclidean distance in it is a usable stand-in for perceived
colour difference. The conversion is in `colorization/color.py` and matches the
published values for the sRGB primaries to two decimals.

**Bins.** The `ab` plane is cut into 10 by 10 Lab unit cells, and the cells
carrying fewer than 500 of the training pixels sampled are dropped, which leaves
124 holding 99.9% of them. That grid is not what limits the vividness of the
output: quantising the ground truth itself to those 124 centres scores 109.4% of
the truth's own chroma, above it rather than below.

Encoding a pixel to its bin goes through a lookup table rather than a distance
matrix. On two million real pixels it picks the same centre as the exact
calculation 97.2% of the time, and the ones it misses land 0.003% further away
on average. Both figures come from `scripts/bins_check.py`.

**Class labels.** CIFAR-10 comes with them, but the colourisers never see them.
They appear only in the per-class analysis, where they are used to group test
images by how ambiguous their colour is.

## Report

The two page report is in `report/`, along with the figures it uses.
