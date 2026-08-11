"""Backward-compatible shim. The real implementation now lives in the
installable `humi` package (see ../humi/_mixture.py). Kept so the
experiment/reproduction scripts in this folder keep working unmodified.
"""

from humi._mixture import mixture_mi, nmi as NMI

__all__ = ["mixture_mi", "NMI"]
