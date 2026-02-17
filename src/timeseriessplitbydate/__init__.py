"""Public package interface for timeseriessplitbydate."""

__version__ = "0.1.0"

from .splitter import TimeSeriesSplitByDate
from .synthetic import make_synthetic_time_series_dataset

__all__ = ["TimeSeriesSplitByDate", "make_synthetic_time_series_dataset", "__version__"]
