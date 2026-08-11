"""Small synthetic-data helpers, mainly useful for quickstart examples and
smoke tests."""

import numpy as np


def synthetic_mixture(num_samples=1000, seed=None):
    """Sample (event, signal) pairs where the signal is a mixture of a
    discrete component (repeated values) and a continuous (Gaussian)
    component, conditioned on a binary event.

    Returns
    -------
    events : np.ndarray of shape (num_samples,)
    signal : np.ndarray of shape (num_samples,)
    """
    rng = np.random.default_rng(seed)
    discrete_y = [-1, 0, 1]
    p_discrete = 0.5

    events = np.zeros(num_samples)
    signal = np.zeros(num_samples)
    for i in range(num_samples):
        if rng.uniform(0, 1) < p_discrete:
            signal[i] = discrete_y[rng.integers(0, len(discrete_y))]
            events[i] = 1
        else:
            signal[i] = rng.normal(0, 1)
            events[i] = 0
    return events, signal
