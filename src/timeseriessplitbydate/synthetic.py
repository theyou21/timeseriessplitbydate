"""Synthetic dataset helpers for examples and tests."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def make_synthetic_time_series_dataset(
    n_samples: int = 48,
    *,
    seed: int = 42,
    shuffle: bool = True,
    date_col: str = "event_timestamp",
) -> pd.DataFrame:
    """Create a synthetic time-series dataset with uneven timestamps.

    Parameters
    ----------
    n_samples : int, default=48
        Number of rows to generate. Must be at least 6.
    seed : int, default=42
        Random seed used for reproducible noise and row shuffling.
    shuffle : bool, default=True
        If ``True``, shuffle rows after generation so row order is not
        chronological.
    date_col : str, default="event_timestamp"
        Name of the datetime column in the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ``date_col``, ``feature``, and ``target`` columns.
    """
    if not isinstance(n_samples, Integral):
        raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}.")
    if n_samples < 6:
        raise ValueError(f"n_samples must be at least 6, got {n_samples}.")
    if not isinstance(seed, Integral):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}.")
    if not isinstance(shuffle, bool):
        raise TypeError(f"shuffle must be a boolean, got {type(shuffle).__name__}.")
    if not isinstance(date_col, str):
        raise TypeError(f"date_col must be a string, got {type(date_col).__name__}.")
    if not date_col.strip():
        raise ValueError("date_col cannot be empty.")

    rng = np.random.default_rng(int(seed))

    base_steps = np.resize(np.array([1, 2, 1, 3], dtype=np.int64), n_samples)
    day_steps = base_steps + rng.integers(0, 2, size=n_samples, dtype=np.int64)
    cumulative_days = np.cumsum(day_steps)
    hour_offsets = rng.integers(0, 18, size=n_samples, dtype=np.int64)

    start = pd.Timestamp("2024-01-01 08:00:00")
    timestamps = start + pd.to_timedelta(cumulative_days, unit="D")
    timestamps = timestamps + pd.to_timedelta(hour_offsets, unit="h")

    trend = np.linspace(0.0, 1.5, n_samples)
    seasonal = np.sin(np.linspace(0.0, 4.0 * np.pi, n_samples))
    noise = rng.normal(loc=0.0, scale=0.2, size=n_samples)
    feature = trend + seasonal + noise
    target = (feature > np.median(feature)).astype(np.int64)

    data = pd.DataFrame(
        {
            date_col.strip(): timestamps,
            "feature": feature,
            "target": target,
        }
    )

    if shuffle:
        data = data.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    return data
