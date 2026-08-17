"""Redundancy-aware MI: clusters similar event states before estimating MI,
so repeated or highly overlapping event types do not fragment the data or
overweight the final score.
"""

from collections import defaultdict

import numpy as np
from scipy.stats import energy_distance
from sklearn.cluster import AgglomerativeClustering

from . import _mixture


def _auto_cluster_by_threshold(distance_matrix, percentile):
    d = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    return np.percentile(d, percentile)


def _merge_groups(groups, labels):
    new_groups = defaultdict(list)
    for g in groups:
        new_groups[labels[g]] += groups[g]
    return new_groups


def normalized_clustered_mi(discs, conts, percentile=90):
    """Normalized MI between a discrete event array and a continuous signal
    array, with redundant/highly correlated event states merged into latent
    clusters before estimation."""
    groups = {}

    size = len(discs)
    event2id = defaultdict(int)
    ids = 0
    for i in range(size):
        if discs[i] not in event2id:
            event2id[discs[i]] = ids
            ids += 1
        label = event2id[discs[i]]
        groups.setdefault(label, []).append(conts[i])

    n = len(groups)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = energy_distance(np.asarray(groups[i]), np.asarray(groups[j]))

    threshold = _auto_cluster_by_threshold(distance_matrix, percentile=percentile)

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="precomputed", linkage="complete"
    )
    labels = clustering.fit_predict(distance_matrix)

    new_groups = _merge_groups(groups, labels)

    new_discs = []
    new_conts = []
    for g in new_groups:
        for val in new_groups[g]:
            new_discs.append(g)
            new_conts.append(val)

    return _mixture.nmi(new_discs, new_conts)
