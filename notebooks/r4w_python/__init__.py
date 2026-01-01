"""R4W Python Wrapper

Python interface to the R4W CLI for use in Jupyter notebooks.
"""

from .cli import R4W
from .plotting import (
    plot_constellation,
    plot_spectrum,
    plot_waterfall,
    plot_time_domain,
    plot_ber_curve,
)

__version__ = "0.1.0"
__all__ = [
    "R4W",
    "plot_constellation",
    "plot_spectrum",
    "plot_waterfall",
    "plot_time_domain",
    "plot_ber_curve",
]
