"""Date-aware cross-validation splitter compatible with scikit-learn workflows."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.utils.validation import indexable

_SplitBy = Literal["days", "weeks", "months"]
_WeekStart = Literal["monday", "sunday"]


@dataclass(frozen=True)
class _ValidatedParams:
    n_splits: int
    date_col: str | None
    max_train_size: int | None
    test_size: int | None
    gap: int
    split_by: _SplitBy
    week_start: _WeekStart


class TimeSeriesSplitByDate(BaseEstimator, BaseCrossValidator):
    """Time-series cross-validator with optional date-column-aware splitting.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds used by ``split_by='days'`` and fallback
        ``TimeSeriesSplit`` behavior.
    date_col : str or None, default=None
        Name of the datetime column in ``X`` when ``X`` is a pandas DataFrame.
        If ``None``, the splitter can still use dates provided through
        :meth:`set_date_data`.
    max_train_size : int or None, default=None
        Maximum number of samples to keep in each training fold.
    test_size : int or None, default=None
        Maximum number of samples to keep in each testing fold.
    gap : int, default=0
        Gap between training and testing windows.

        - For date-based splitting, interpreted as calendar days.
        - For fallback ``TimeSeriesSplit``, interpreted in samples.
    split_by : {"days", "weeks", "months"}, default="days"
        Date-based splitting strategy.

        - ``"days"`` divides the observed date span into equal-width intervals.
        - ``"weeks"`` creates adjacent week folds based on ``week_start``
          (train through week ``i``, test on week ``i+1``).
        - ``"months"`` creates adjacent month folds (train through month ``i``,
          test on month ``i+1``).
    week_start : {"monday", "sunday"}, default="monday"
        First day of the week used when ``split_by='weeks'``.
        ``"monday"`` is the default because it is the most common convention in
        analytics tooling (ISO week standard).

    Notes
    -----
    This splitter follows scikit-learn conventions:

    - ``split(X, y=None, groups=None)`` and ``get_n_splits(X=None, ...)`` are
      supported.
    - If no date source is available, it falls back to
      :class:`sklearn.model_selection.TimeSeriesSplit`.
    - ``groups`` is accepted for compatibility and ignored.

    Date source precedence is:

    1. Data provided with :meth:`set_date_data`
    2. ``date_col`` from a pandas DataFrame ``X``
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        date_col: str | None = None,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
        split_by: _SplitBy = "days",
        week_start: _WeekStart = "monday",
    ) -> None:
        # Keep constructor side-effect free per scikit-learn estimator conventions.
        self.n_splits = n_splits
        self.date_col = date_col
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.split_by = split_by
        self.week_start = week_start

        self._date_data: pd.Series | None = None

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : array-like or DataFrame, default=None
            Input data. Used for month-based mode when ``date_col`` is set.
        y : array-like, default=None
            Ignored, present for API compatibility.
        groups : array-like, default=None
            Ignored, present for API compatibility.

        Returns
        -------
        int
            Number of splits.

            For ``split_by='weeks'`` or ``split_by='months'`` with available
            date data, this is the number of unique periods minus one.
        """
        del y, groups

        params = self._validated_params()

        if params.split_by == "days":
            return params.n_splits

        try:
            resolved_dates = self._resolve_dates(
                X=X,
                n_samples=None,
                date_col=params.date_col,
            )
        except (TypeError, ValueError):
            # Keep get_n_splits robust when date context is unavailable.
            return params.n_splits
        if resolved_dates is None:
            return params.n_splits

        if params.split_by == "weeks":
            freq = self._week_period_freq(params.week_start)
            unique_weeks = resolved_dates.dt.to_period(freq).nunique()
            return max(unique_weeks - 1, 0)

        unique_months = resolved_dates.dt.to_period("M").nunique()
        return max(unique_months - 1, 0)

    def set_date_data(self, date_data: Iterable[Any]) -> TimeSeriesSplitByDate:
        """Store external date values to use in subsequent :meth:`split` calls.

        Parameters
        ----------
        date_data : iterable
            Datetime-like values aligned with samples passed to :meth:`split`.

        Returns
        -------
        TimeSeriesSplitByDate
            The current instance.
        """
        self._date_data = self._coerce_dates(date_data, source="date_data")
        return self

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for API compatibility.
        groups : array-like of shape (n_samples,), default=None
            Group labels. Ignored, present for API compatibility.

        Yields
        ------
        train : ndarray of shape (n_train_samples,)
            Training set indices for the split.
        test : ndarray of shape (n_test_samples,)
            Testing set indices for the split.

        Notes
        -----
        If no date source is available (no ``set_date_data`` and no ``date_col``),
        this method falls back to :class:`sklearn.model_selection.TimeSeriesSplit`.
        """
        params = self._validated_params()
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)

        if groups is not None:
            warnings.warn(
                f"The 'groups' parameter is ignored by {self.__class__.__name__}.",
                UserWarning,
                stacklevel=2,
            )

        resolved_dates = self._resolve_dates(
            X=X,
            n_samples=n_samples,
            date_col=params.date_col,
        )
        if resolved_dates is None:
            fallback_cv = TimeSeriesSplit(
                n_splits=params.n_splits,
                max_train_size=params.max_train_size,
                test_size=params.test_size,
                gap=params.gap,
            )
            yield from fallback_cv.split(X, y, groups)
            return

        if params.split_by == "months":
            yield from self._split_by_months(
                dates=resolved_dates,
                gap=params.gap,
                max_train_size=params.max_train_size,
                test_size=params.test_size,
            )
            return

        if params.split_by == "weeks":
            yield from self._split_by_weeks(
                dates=resolved_dates,
                gap=params.gap,
                max_train_size=params.max_train_size,
                test_size=params.test_size,
                week_start=params.week_start,
            )
            return

        yield from self._split_by_days(
            dates=resolved_dates,
            n_splits=params.n_splits,
            gap=params.gap,
            max_train_size=params.max_train_size,
            test_size=params.test_size,
        )

    def _validated_params(self) -> _ValidatedParams:
        return _ValidatedParams(
            n_splits=self._validate_n_splits(self.n_splits),
            date_col=self._validate_date_col(self.date_col),
            max_train_size=self._validate_optional_positive_int(
                self.max_train_size,
                name="max_train_size",
            ),
            test_size=self._validate_optional_positive_int(
                self.test_size,
                name="test_size",
            ),
            gap=self._validate_non_negative_int(self.gap, name="gap"),
            split_by=self._validate_split_by(self.split_by),
            week_start=self._validate_week_start(self.week_start),
        )

    @staticmethod
    def _validate_n_splits(n_splits: int) -> int:
        if not isinstance(n_splits, Integral):
            raise TypeError(f"n_splits must be an integer, got {type(n_splits).__name__}.")
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
        return int(n_splits)

    @staticmethod
    def _validate_date_col(date_col: str | None) -> str | None:
        if date_col is None:
            return None
        if not isinstance(date_col, str):
            raise TypeError(f"date_col must be a string or None, got {type(date_col).__name__}.")
        stripped = date_col.strip()
        if not stripped:
            raise ValueError("date_col cannot be an empty string.")
        return stripped

    @staticmethod
    def _validate_optional_positive_int(value: int | None, *, name: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer or None, got {type(value).__name__}.")
        if value < 1:
            raise ValueError(f"{name} must be >= 1 when provided, got {value}.")
        return int(value)

    @staticmethod
    def _validate_non_negative_int(value: int, *, name: str) -> int:
        if not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
        return int(value)

    @staticmethod
    def _validate_split_by(split_by: str) -> _SplitBy:
        if split_by not in {"days", "weeks", "months"}:
            raise ValueError(
                "split_by must be one of {'days', 'weeks', 'months'}, "
                f"got {split_by!r}."
            )
        return split_by

    @staticmethod
    def _validate_week_start(week_start: str) -> _WeekStart:
        if not isinstance(week_start, str):
            raise TypeError(
                f"week_start must be a string, got {type(week_start).__name__}."
            )
        normalized = week_start.strip().lower()
        if normalized not in {"monday", "sunday"}:
            raise ValueError(
                "week_start must be either 'monday' or 'sunday', "
                f"got {week_start!r}."
            )
        return normalized

    @staticmethod
    def _week_period_freq(week_start: _WeekStart) -> str:
        # Pandas weekly periods are defined by week end day.
        return "W-SUN" if week_start == "monday" else "W-SAT"

    def _resolve_dates(
        self,
        *,
        X: Any,
        n_samples: int | None,
        date_col: str | None,
    ) -> pd.Series | None:
        source: str | None = None
        candidate: Iterable[Any] | None = None

        if self._date_data is not None:
            source = "date_data"
            candidate = self._date_data
        elif date_col is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    "date_col is set, so X must be a pandas DataFrame. "
                    "Alternatively call set_date_data(...)."
                )
            if date_col not in X.columns:
                raise ValueError(
                    f"date_col='{date_col}' was not found in DataFrame columns: "
                    f"{list(X.columns)!r}."
                )
            source = f"X['{date_col}']"
            candidate = X[date_col]

        if candidate is None:
            return None

        resolved = self._coerce_dates(candidate, source=source or "date_data")
        if n_samples is not None and len(resolved) != n_samples:
            raise ValueError(
                f"Date data length ({len(resolved)}) must match number "
                f"of samples in X ({n_samples})."
            )
        return resolved

    @staticmethod
    def _coerce_dates(values: Iterable[Any], *, source: str) -> pd.Series:
        try:
            resolved = pd.to_datetime(pd.Series(values), errors="coerce")
        except Exception as exc:  # pragma: no cover - defensive conversion guard
            raise ValueError(f"Failed to parse datetime values from {source}.") from exc

        invalid_count = int(resolved.isna().sum())
        if invalid_count:
            raise ValueError(
                f"{source} contains {invalid_count} missing or invalid datetime value(s)."
            )
        return resolved.reset_index(drop=True)

    def _split_by_months(
        self,
        *,
        dates: pd.Series,
        gap: int,
        max_train_size: int | None,
        test_size: int | None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        unique_months = sorted(dates.dt.to_period("M").unique())
        if len(unique_months) < 2:
            raise ValueError(
                "Month-based splitting requires at least two distinct months in the date data."
            )

        yielded = 0
        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_end = train_month.to_timestamp(how="end")
            test_start = test_month.to_timestamp(how="start") + pd.Timedelta(days=gap)
            test_end = test_month.to_timestamp(how="end")

            train_candidates = np.flatnonzero((dates <= train_end).to_numpy())
            test_candidates = np.flatnonzero(
                ((dates >= test_start) & (dates <= test_end)).to_numpy()
            )

            train_indices = self._apply_max_train_size(
                indices=train_candidates,
                dates=dates,
                max_train_size=max_train_size,
            )
            test_indices = self._apply_test_size(
                indices=test_candidates,
                dates=dates,
                test_size=test_size,
            )

            if train_indices.size > 0 and test_indices.size > 0:
                yielded += 1
                yield train_indices, test_indices

        if yielded == 0:
            raise ValueError(
                "No valid splits were generated for split_by='months'. "
                "Try reducing gap, max_train_size, or test_size."
            )

    def _split_by_weeks(
        self,
        *,
        dates: pd.Series,
        gap: int,
        max_train_size: int | None,
        test_size: int | None,
        week_start: _WeekStart,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        freq = self._week_period_freq(week_start)
        unique_weeks = sorted(dates.dt.to_period(freq).unique())
        if len(unique_weeks) < 2:
            raise ValueError(
                "Week-based splitting requires at least two distinct weeks in the date data."
            )

        yielded = 0
        for i in range(len(unique_weeks) - 1):
            train_week = unique_weeks[i]
            test_week = unique_weeks[i + 1]

            train_end = train_week.to_timestamp(how="end")
            test_start = test_week.to_timestamp(how="start") + pd.Timedelta(days=gap)
            test_end = test_week.to_timestamp(how="end")

            train_candidates = np.flatnonzero((dates <= train_end).to_numpy())
            test_candidates = np.flatnonzero(
                ((dates >= test_start) & (dates <= test_end)).to_numpy()
            )

            train_indices = self._apply_max_train_size(
                indices=train_candidates,
                dates=dates,
                max_train_size=max_train_size,
            )
            test_indices = self._apply_test_size(
                indices=test_candidates,
                dates=dates,
                test_size=test_size,
            )

            if train_indices.size > 0 and test_indices.size > 0:
                yielded += 1
                yield train_indices, test_indices

        if yielded == 0:
            raise ValueError(
                "No valid splits were generated for split_by='weeks'. "
                "Try reducing gap, max_train_size, or test_size."
            )

    def _split_by_days(
        self,
        *,
        dates: pd.Series,
        n_splits: int,
        gap: int,
        max_train_size: int | None,
        test_size: int | None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        min_date = dates.min()
        max_date = dates.max()
        if min_date == max_date:
            raise ValueError("Day-based splitting requires at least two distinct datetime values.")

        date_range = max_date - min_date
        interval_size = date_range / (n_splits + 1)

        yielded = 0
        for i in range(n_splits):
            train_end = min_date + interval_size * (i + 1)
            test_start = train_end.normalize() + pd.Timedelta(days=gap + 1)
            test_end = min_date + interval_size * (i + 2)

            train_candidates = np.flatnonzero((dates <= train_end).to_numpy())
            test_candidates = np.flatnonzero(
                ((dates >= test_start) & (dates <= test_end)).to_numpy()
            )

            train_indices = self._apply_max_train_size(
                indices=train_candidates,
                dates=dates,
                max_train_size=max_train_size,
            )
            test_indices = self._apply_test_size(
                indices=test_candidates,
                dates=dates,
                test_size=test_size,
            )

            if train_indices.size > 0 and test_indices.size > 0:
                yielded += 1
                yield train_indices, test_indices

        if yielded == 0:
            raise ValueError(
                "No valid splits were generated for split_by='days'. "
                "Try reducing n_splits, gap, max_train_size, or test_size."
            )

    @staticmethod
    def _apply_max_train_size(
        *,
        indices: np.ndarray,
        dates: pd.Series,
        max_train_size: int | None,
    ) -> np.ndarray:
        if max_train_size is None or indices.size <= max_train_size:
            return np.asarray(indices, dtype=np.int64)

        ordered = indices[np.argsort(dates.iloc[indices].to_numpy())]
        trimmed = ordered[-max_train_size:]
        return np.sort(trimmed.astype(np.int64, copy=False))

    @staticmethod
    def _apply_test_size(
        *,
        indices: np.ndarray,
        dates: pd.Series,
        test_size: int | None,
    ) -> np.ndarray:
        if test_size is None or indices.size <= test_size:
            return np.asarray(indices, dtype=np.int64)

        ordered = indices[np.argsort(dates.iloc[indices].to_numpy())]
        trimmed = ordered[:test_size]
        return np.sort(trimmed.astype(np.int64, copy=False))
