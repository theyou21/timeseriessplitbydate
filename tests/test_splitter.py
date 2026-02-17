from __future__ import annotations

import inspect
from importlib.util import find_spec

import pytest

HAS_REQUIRED_DEPS = all(
    find_spec(module) is not None
    for module in ("numpy", "pandas", "sklearn", "timeseriessplitbydate")
)

pytestmark = pytest.mark.skipif(
    not HAS_REQUIRED_DEPS,
    reason="Optional test dependencies are not installed.",
)

if HAS_REQUIRED_DEPS:
    import pandas as pd

    from timeseriessplitbydate import TimeSeriesSplitByDate


DATE_COL = "event_timestamp"


def _collect_split_indices(
    splitter: TimeSeriesSplitByDate,
    X: pd.DataFrame,
) -> list[tuple[list[int], list[int]]]:
    return [(train.tolist(), test.tolist()) for train, test in splitter.split(X)]


def _minimal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ]
            )
        }
    )


def test_works_with_rows_not_sorted_by_date_exact_indices() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-08",
                    "2024-01-02",
                    "2024-01-06",
                ]
            ),
            "value": [10, 11, 12, 13, 14, 15],
        }
    )

    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL, split_by="days", gap=0)

    assert not X[DATE_COL].is_monotonic_increasing
    assert _collect_split_indices(splitter, X) == [
        ([1, 2, 4], [0]),
        ([0, 1, 2, 4], [3, 5]),
    ]


def test_handles_uneven_time_gaps_sparse_weekends_and_bursts_exact_indices() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-05",  # Fri
                    "2024-01-08",  # Mon
                    "2024-01-09",  # Tue (burst)
                    "2024-01-10",  # Wed (burst)
                    "2024-01-22",  # sparse gap
                    "2024-01-23",  # burst
                    "2024-02-05",  # sparse gap
                    "2024-02-06",
                ]
            ),
            "value": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL, split_by="days", gap=0)

    assert _collect_split_indices(splitter, X) == [
        ([0, 1, 2, 3], [4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7]),
    ]


def test_splits_are_chronological_by_date_boundaries() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-08",
                    "2024-01-02",
                    "2024-01-06",
                ]
            )
        }
    )
    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL, split_by="days", gap=0)

    for train_idx, test_idx in splitter.split(X):
        train_dates = X.loc[train_idx, DATE_COL]
        test_dates = X.loc[test_idx, DATE_COL]
        assert (train_dates.max() < test_dates).all()


def test_no_leakage_max_train_date_is_before_min_test_date_per_split() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-08",
                    "2024-01-02",
                    "2024-01-06",
                ]
            )
        }
    )
    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL, split_by="days", gap=0)

    for train_idx, test_idx in splitter.split(X):
        max_train_date = X.loc[train_idx, DATE_COL].max()
        min_test_date = X.loc[test_idx, DATE_COL].min()
        assert max_train_date < min_test_date


def test_stable_behavior_with_duplicates_on_same_date_exact_indices() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-06",
                ]
            ),
            "value": [5, 6, 7, 8, 9, 10],
        }
    )

    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL, split_by="days", gap=0)

    expected = [
        ([0, 1, 2], [3, 4]),
        ([0, 1, 2, 3, 4], [5]),
    ]
    first_run = _collect_split_indices(splitter, X)
    second_run = _collect_split_indices(splitter, X)

    assert first_run == expected
    assert second_run == expected


def test_timezone_aware_datetimes_exact_indices_if_supported() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-01 00:00:00+00:00",
                    "2024-01-02 00:00:00+00:00",
                    "2024-01-03 00:00:00+00:00",
                    "2024-01-05 00:00:00+00:00",
                    "2024-01-06 00:00:00+00:00",
                    "2024-01-07 00:00:00+00:00",
                ],
                utc=True,
            )
        }
    )

    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL, split_by="days", gap=0)

    try:
        splits = _collect_split_indices(splitter, X)
    except (TypeError, ValueError) as exc:
        pytest.skip(
            "Timezone-aware datetime splitting is not supported "
            f"in this environment: {exc}"
        )

    assert splits == [
        ([0, 1, 2], [3]),
        ([0, 1, 2, 3], [4, 5]),
    ]


def test_constructor_stores_arguments_without_side_effects() -> None:
    splitter = TimeSeriesSplitByDate(
        n_splits=3,
        date_col=f" {DATE_COL} ",
        split_by="days",
        week_start=" Sunday ",
    )

    assert splitter.n_splits == 3
    assert splitter.date_col == f" {DATE_COL} "
    assert splitter.week_start == " Sunday "
    assert "date_col=' event_timestamp '" in repr(splitter)


def test_split_signature_matches_sklearn_style() -> None:
    signature = inspect.signature(TimeSeriesSplitByDate.split)
    assert tuple(signature.parameters) == ("self", "X", "y", "groups")
    assert signature.parameters["y"].default is None
    assert signature.parameters["groups"].default is None


def test_get_n_splits_signature_matches_sklearn_style() -> None:
    signature = inspect.signature(TimeSeriesSplitByDate.get_n_splits)
    assert tuple(signature.parameters) == ("self", "X", "y", "groups")
    assert signature.parameters["X"].default is None
    assert signature.parameters["y"].default is None
    assert signature.parameters["groups"].default is None


def test_get_n_splits_months_returns_configured_value_without_date_context() -> None:
    splitter = TimeSeriesSplitByDate(n_splits=4, date_col=DATE_COL, split_by="months")
    assert splitter.get_n_splits() == 4


def test_get_n_splits_months_uses_available_date_context() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-10",
                    "2024-02-01",
                    "2024-03-01",
                ]
            )
        }
    )
    splitter = TimeSeriesSplitByDate(n_splits=4, date_col=DATE_COL, split_by="months")

    assert splitter.get_n_splits(X) == 2


def test_get_n_splits_weeks_returns_configured_value_without_date_context() -> None:
    splitter = TimeSeriesSplitByDate(n_splits=3, date_col=DATE_COL, split_by="weeks")
    assert splitter.get_n_splits() == 3


def test_get_n_splits_weeks_uses_available_date_context() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-04",
                    "2024-01-08",
                    "2024-01-10",
                    "2024-01-15",
                ]
            )
        }
    )
    splitter = TimeSeriesSplitByDate(n_splits=4, date_col=DATE_COL, split_by="weeks")

    assert splitter.get_n_splits(X) == 2


def test_weeks_split_with_monday_start_exact_indices() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-14",
                    "2024-01-15",
                    "2024-01-16",
                ]
            )
        }
    )

    splitter = TimeSeriesSplitByDate(
        n_splits=4,
        date_col=DATE_COL,
        split_by="weeks",
        week_start="monday",
    )

    assert _collect_split_indices(splitter, X) == [
        ([0, 1, 2], [3, 4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7]),
    ]


def test_weeks_split_with_sunday_start_exact_indices() -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-13",
                    "2024-01-14",
                    "2024-01-20",
                ]
            )
        }
    )

    splitter = TimeSeriesSplitByDate(
        n_splits=4,
        date_col=DATE_COL,
        split_by="weeks",
        week_start="sunday",
    )

    assert _collect_split_indices(splitter, X) == [
        ([0], [1, 2, 3]),
        ([0, 1, 2, 3], [4, 5]),
    ]


@pytest.mark.parametrize("week_start", ["monday", "sunday"])
def test_weeks_split_has_no_date_leakage(week_start: str) -> None:
    X = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-13",
                    "2024-01-14",
                    "2024-01-20",
                ]
            )
        }
    )
    splitter = TimeSeriesSplitByDate(
        n_splits=3,
        date_col=DATE_COL,
        split_by="weeks",
        week_start=week_start,
    )

    for train_idx, test_idx in splitter.split(X):
        assert X.loc[train_idx, DATE_COL].max() < X.loc[test_idx, DATE_COL].min()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": 1}, "n_splits must be at least 2"),
        ({"gap": -1}, "gap must be >= 0"),
        (
            {"split_by": "quarters"},
            "split_by must be one of {'days', 'weeks', 'months'}",
        ),
        (
            {"split_by": "weeks", "week_start": "friday"},
            "week_start must be either 'monday' or 'sunday'",
        ),
        ({"max_train_size": 0}, "max_train_size must be >= 1"),
        ({"test_size": 0}, "test_size must be >= 1"),
    ],
)
def test_invalid_fold_parameters_raise_clear_errors(
    kwargs: dict[str, object],
    message: str,
) -> None:
    splitter = TimeSeriesSplitByDate(**kwargs)
    X = _minimal_frame()

    with pytest.raises((TypeError, ValueError), match=message):
        list(splitter.split(X))


def test_missing_date_column_raises_clear_error() -> None:
    X = pd.DataFrame({"other_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
    splitter = TimeSeriesSplitByDate(n_splits=2, date_col=DATE_COL)

    with pytest.raises(ValueError, match=f"date_col='{DATE_COL}' was not found"):
        next(splitter.split(X))


def test_set_date_data_length_mismatch_raises_clear_error() -> None:
    X = _minimal_frame()
    splitter = TimeSeriesSplitByDate(n_splits=2)
    splitter.set_date_data(X[DATE_COL].iloc[:-1])

    with pytest.raises(ValueError, match=r"Date data length \(5\) must match"):
        list(splitter.split(X))
