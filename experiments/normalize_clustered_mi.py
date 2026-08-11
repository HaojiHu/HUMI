"""Backward-compatible shim. The real implementation now lives in the
installable `humi` package (see ../humi/_cluster.py). Kept so the
experiment/reproduction scripts in this folder keep working unmodified.
"""

from humi._cluster import normalized_clustered_mi as NormalizedClusteredMI

__all__ = ["NormalizedClusteredMI"]
