# Reproducing the paper's experiments

This folder reproduces the results in *Estimating Mutual Information between
Time Series and Temporal Event Sequences Across Diverse Analysis Tasks* (KDD
2026). The estimator itself now lives in the installable [`humi`](../humi)
package; everything here is paper-specific analysis and baseline code.

Install the core package plus the extra dependencies these scripts need
(forecasting baselines, foundation models, notebooks, etc.):

```bash
pip install -e ..
pip install -r requirements.txt
```

`Covariate_selection_rossmann.py` and `Covariate_selection_M5.py` also need
[TimesFM](https://github.com/google-research/timesfm) and
[Chronos-2](https://huggingface.co/amazon/chronos-2).

## Layout

**Time-delayed mutual information (causality / lag)**
- `tdmi_controlled_digit_ground_truth_exp.py`
- `tdmi_controlled_digit_synthetic_exp.py`, `tdmi_synthetic.py`

**Repeated temporal pattern**
- Seasonality: `Autocorrelation.py`, `Fourier_transform.py`, `seasonality_real_data.py`
- Local/point repeated pattern: `tempurature_exp.py`

**Continuous feature selection**
- `feature_selection.py`, `variants_for_ablation_study.py`

**Discrete covariate selection**
- `Covariate_selection_rossmann.py`, `Covariate_selection_M5.py`

**Baselines and helpers**
- `ross_mi.py`, `mixed.py`, `mi_mao.py`, `mixture_mi_mao.py` — comparison estimators
- `DeepAD.py`, `TimeSeriesFoundationModel.py`, `fm_time_series.py`, `Chronos2.py`, `CatBoostRegressor*.py`, `ARIMA.py` — forecasting baselines
- `normalize_clustered_mi.py`, `mixture_mi.py`, `entropy_mao.py`, `mixture_dataset.py` — thin shims re-exporting `humi`, kept so the scripts above run unmodified

The continuous feature selection and discrete covariate selection experiments
take a while to run, so their detailed results are also saved for quick
checking:

- `feature_selection_threshold90.txt`
- `M5_threshold90.txt`, `rossmann_threshold90.txt`

If you use this code, please cite the paper (see the [main README](../README.md#citation)).
