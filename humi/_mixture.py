"""Mixture-variable mutual information: MI between a discrete variable and a
continuous variable whose empirical distribution mixes a continuous component
with a precision-induced discrete (repeated-value) component.
"""

import math
from collections import defaultdict

import numpy as np

from ._entropy import entropy_estimate_1d_continuous, entropy_estimate_1d_discrete


def _obtain_repeated_values(time_series):
    val2freq = defaultdict(int)
    for val in time_series:
        val2freq[val] += 1

    for val in list(val2freq.keys()):
        if val2freq[val] == 1:
            del val2freq[val]

    return val2freq


def _partition_data_pairs(disc, cont):
    val2freq = _obtain_repeated_values(cont)
    dis_dis = []
    dis_cont = []
    for i, val in enumerate(cont):
        if val in val2freq:
            dis_dis.append([disc[i], val])
        else:
            dis_cont.append([disc[i], val])

    dis_dis = tuple(np.asarray(row) for row in zip(*dis_dis))
    dis_cont = tuple(np.asarray(row) for row in zip(*dis_cont))

    return dis_dis, dis_cont, len(val2freq)


def mixture_mi(disc, cont):
    """Un-normalized MI between a discrete array and a continuous-discrete
    mixture array, using the continuous-discrete duality representation."""
    dis_dis, dis_cont, _ = _partition_data_pairs(disc, cont)
    if len(dis_dis) == 0:
        H_Y_A = 0
    else:
        H_Y_A = entropy_estimate_1d_discrete(dis_dis[1])

    H_XY_A = 0
    groups = dict()
    if len(dis_dis) > 0:
        for x, y in zip(dis_dis[0], dis_dis[1]):
            if (x, y) in groups:
                groups[(x, y)] += 1
            else:
                groups[(x, y)] = 1

        for k in groups.keys():
            p = groups[k] / len(dis_dis[0])
            H_XY_A += -p * np.log(p)

    if len(dis_cont) > 1 and len(dis_cont[1]) > 1:
        H_Y_B = entropy_estimate_1d_continuous(dis_cont[1])
    else:
        H_Y_B = 0

    if len(dis_dis) == 0:
        p_a = 0
    else:
        p_a = len(dis_dis[0]) / len(disc)

    H_XY_B = 0
    if len(dis_cont) > 1:
        discrete_range = set(dis_cont[0])
        groups = defaultdict(np.array)
        for d in discrete_range:
            groups[d] = dis_cont[1][dis_cont[0] == d]
        for g in groups.keys():
            if len(groups[g]) > 1:
                H_XY_B += len(groups[g]) / (len(dis_cont[0])) * entropy_estimate_1d_continuous(groups[g], islog2=False)
                H_XY_B -= len(groups[g]) / (len(dis_cont[0])) * np.log(len(groups[g]) / (len(dis_cont[0])))

    H_X = entropy_estimate_1d_discrete(disc)

    p_b = 1 - p_a
    MI = p_a * (H_Y_A - H_XY_A) + p_b * (H_Y_B - H_XY_B) + H_X

    return max(MI, 0)


def nmi(discs, conts, normalized=True):
    """Mutual information between a discrete event array and a continuous
    (possibly duplicate-valued) signal array, optionally normalized to
    [0, 1]."""
    conts = np.asarray(conts, dtype=float)
    scale = np.max(conts) - np.min(conts)
    conts = (conts - np.min(conts)) / scale

    unit = 1
    for i in range(len(conts) - 1):
        if unit > conts[i] - np.min(conts) > 0:
            unit = conts[i] - np.min(conts)
    unit = unit / len(conts)

    mi = mixture_mi(discs, conts)

    if not normalized:
        return mi
    return math.sqrt(1 - math.exp((-2 * mi) / (unit + 1)))
