"""HUMI: Mutual Information between Time Series and Temporal Event Sequence.

>>> from humi import estimate_mi
>>> from humi.datasets import synthetic_mixture
>>> events, signal = synthetic_mixture(1000, seed=0)
>>> estimate_mi(events, signal)
"""

from ._cluster import normalized_clustered_mi
from ._mixture import nmi

__version__ = "0.1.0"
__all__ = ["estimate_mi"]


def estimate_mi(events, signal, cluster=True, percentile=90, normalized=True):
    """Estimate the mutual information between a discrete temporal event
    sequence and a continuous time series signal, aligned pairwise.

    Parameters
    ----------
    events : sequence
        Discrete event state for each aligned time step (e.g. weekday,
        holiday, promotion, medication status).
    signal : sequence of float
        Continuous signal value for each aligned time step (e.g. traffic
        volume, sales, temperature).
    cluster : bool, default True
        If True, merge redundant or highly correlated event states into
        latent clusters before estimating MI (recommended when the event
        vocabulary is large or overlapping).
    percentile : float, default 90
        Distance-threshold percentile used for clustering when
        ``cluster=True``. Ignored otherwise.
    normalized : bool, default True
        If True, return a score bounded to [0, 1]. If False, return the
        raw (nat-scaled) mutual information estimate.

    Returns
    -------
    float
        The estimated (normalized) mutual information score.
    """
    if cluster:
        return normalized_clustered_mi(events, signal, percentile=percentile)
    return nmi(events, signal, normalized=normalized)
