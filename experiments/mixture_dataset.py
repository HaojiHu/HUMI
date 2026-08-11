"""Backward-compatible shim. The real implementation now lives in the
installable `humi` package (see ../humi/datasets.py). Kept so the
experiment/reproduction scripts in this folder keep working unmodified.
"""

import numpy as np

from humi.datasets import synthetic_mixture as get_mixture


def discrete_flag(arr):
    unique_set = set()
    multiple_set = set()
    for e in arr:
        if e in unique_set:
            unique_set.remove(e)
            multiple_set.add(e)
        elif e in multiple_set:
            continue
        else:
            unique_set.add(e)
    discrete = np.ones(len(arr))
    for i, e in enumerate(arr):
        if e in unique_set:
            discrete[i] = 0
    return discrete


def unmix(arrs):
    flag = discrete_flag(arrs[1])
    dis_dis = (arrs[0][flag == 1], arrs[1][flag == 1])
    dis_cont = (arrs[0][flag == 0], arrs[1][flag == 0])
    return dis_dis, dis_cont


__all__ = ["get_mixture", "discrete_flag", "unmix"]
