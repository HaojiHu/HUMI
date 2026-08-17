"""Backward-compatible shim. The real implementation now lives in the
installable `humi` package (see ../humi/_entropy.py). Kept so the
experiment/reproduction scripts in this folder keep working unmodified.
"""

from humi._entropy import entropy_estimate_1d_continuous, entropy_estimate_1d_discrete

__all__ = ["entropy_estimate_1d_continuous", "entropy_estimate_1d_discrete"]
