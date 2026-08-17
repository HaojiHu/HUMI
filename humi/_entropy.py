"""Nonparametric 1D entropy estimators for discrete and continuous samples."""

import numpy as np
from scipy.special import gamma


def entropy_estimate_1d_discrete(arr, islog2=False):
    counter = {}
    tot = 0
    for e in arr:
        if e not in counter:
            counter[e] = 1
        else:
            counter[e] += 1
        tot += 1
    ent = 0
    for k in counter:
        pk = counter[k] / tot
        ent -= pk * np.log(pk)
    ent *= np.log2(np.e) if islog2 else 1.
    return ent


def entropy_estimate_1d_continuous(array, islog2=False):
    arr = np.sort(array)
    size = len(arr)
    rho = np.zeros(size)
    rho[0] = abs(arr[1] - arr[0])
    rho[size - 1] = abs(arr[size - 1] - arr[size - 2])
    for i in range(1, size - 1):
        rho[i] = min(abs(arr[i] - arr[i - 1]), abs(arr[i] - arr[i + 1]))
        if rho[i] == 0:
            rho[i] = max(abs(arr[i] - arr[i - 1]), abs(arr[i] - arr[i + 1]))
    avg_rho = 0
    for i in range(size):
        avg_rho += np.log(rho[i])
    avg_rho = avg_rho * (1. / size)
    ent = avg_rho + np.log(np.pi ** 0.5 / gamma(1. / 2. + 1.)) + 0.5772 + np.log(size - 1)

    ent *= np.log2(np.e) if islog2 else 1.
    return ent
