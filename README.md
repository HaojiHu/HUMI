# HUMI

[![arXiv](https://img.shields.io/badge/arXiv-2606.01602-b31b1b.svg)](https://arxiv.org/abs/2606.01602)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**HUMI** (Mutual Information between Time Series and Temporal Event Sequence)
directly estimates the dependence between a continuous time series and a
discrete temporal event sequence — no discretization, no training, no
learned cross-modal representation. It plays the same role Pearson
correlation plays for two time series, but for heterogeneous data types.

Paper: [Estimating Mutual Information between Time Series and Temporal Event
Sequences Across Diverse Analysis Tasks](https://arxiv.org/abs/2606.01602)
(KDD 2026)

## Install

```bash
git clone https://github.com/HaojiHu/HUMI.git
cd HUMI
pip install -e .
```

Requires Python >= 3.9. Core dependencies are just `numpy`, `scipy` and
`scikit-learn`.

## Quickstart

```python
from humi import estimate_mi
from humi.datasets import synthetic_mixture

events, signal = synthetic_mixture(1000, seed=0)
score = estimate_mi(events, signal)  # normalized MI in [0, 1]
```

In practice, `events` is a discrete state per timestep (weekday, holiday,
promotion, medication status, ...) and `signal` is the aligned continuous
value (traffic volume, sales, temperature, heart rate, ...):

```python
from humi import estimate_mi

score = estimate_mi(events=promotions, signal=sales)
```

`estimate_mi` also accepts:

- `cluster` (default `True`): merge redundant or highly correlated event
  states into latent clusters before estimating, recommended whenever the
  event vocabulary is large or overlapping.
- `percentile` (default `90`): distance-threshold percentile used for that
  clustering.
- `normalized` (default `True`): return a score bounded to `[0, 1]` instead
  of the raw estimate.

## How it works

HUMI combines three ideas:

1. A theoretical MI formulation for one discrete and one continuous
   variable, without forcing either side to become the other.
2. A continuous-discrete duality representation of the signal, so repeated
   values caused by finite measurement precision are modeled directly
   instead of breaking nearest-neighbor entropy estimators.
3. Optional clustering of redundant or highly correlated event states, so
   overlapping event labels don't fragment the data or bias the score.

See the [paper](https://arxiv.org/abs/2606.01602) for the full derivation
and experiments.

## Reproducing the paper

[`experiments/`](experiments/) has the code and data behind every task in
the paper: causality/lag (TDMI), seasonality, local repeated patterns,
discrete covariate selection, and continuous feature selection.

## Citation

```bibtex
@inproceedings{hu2026humi,
  title     = {Estimating Mutual Information between Time Series and Temporal Event Sequences Across Diverse Analysis Tasks},
  author    = {Hu, Haoji and Mao, Huaqing and Lin, Yijun and Jia, Xiaowei and Zhou, Jinwei and Jeong, Minoh and Chiang, Yao-Yi},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```

## License

[MIT](LICENSE)
