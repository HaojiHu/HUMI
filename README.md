# HUMI

[[paper](https://arxiv.org/pdf/2606.01602)]
[[Blog](https://pufferbyte.github.io/kdd26mi/)]

**HUMI** 
directly estimates the dependence between a continuous time series and a
discrete temporal event sequence - no discretization, no training, no
learned cross-modal representation. 

Paper: [Estimating Mutual Information between Time Series and Temporal Event
Sequences Across Diverse Analysis Tasks](https://arxiv.org/abs/2606.01602)
(KDD 2026)

## Install

```bash
git clone https://github.com/HaojiHu/HUMI.git
cd HUMI
pip install -e .
```

Core dependencies are just `numpy`, `scipy` and
`scikit-learn`.

## Getting Started

This example converts a repeated local pattern (the day/night temperature cycle) into a dependence measure between temperature and time-of-day context. DN first labels each timestep as day or night, and then measures how much that context reduces uncertainty about temperature.

```python
import pandas as pd
import humi

# Minneapolis 2023 day-high / night-low temperature
# (experiments/data/tdmi/minneapolis_2023_day_high_night_low.csv)
df = pd.read_csv("experiments/data/tdmi/minneapolis_2023_day_high_night_low.csv")
series = df[["night_low_f", "day_high_f"]].to_numpy().reshape(-1)
events = ["night", "day"] * len(df)

score = humi.humi(events=events, series=series, cluster=False)  # 0.6124
```

See [`experiments/tempurature_exp.py`](experiments/tempurature_exp.py)
for the full reproduction, including the `TwoMon` and `DNTwoMon`
contexts that push the score up to 0.96.

In practice, `events` is a discrete state per timestep (weekday, holiday,
promotion, medication status, ...) and `series` is the aligned continuous
value (traffic volume, sales, temperature, heart rate, ...):

```python
import humi

score = humi.humi(events=promotions, series=sales)
```


`humi` also accepts:

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
