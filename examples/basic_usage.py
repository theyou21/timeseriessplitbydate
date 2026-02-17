"""Basic examples for date-column-based time series splitting."""

from __future__ import annotations

from timeseriessplitbydate import TimeSeriesSplitByDate, make_synthetic_time_series_dataset


def _print_folds(data, date_col: str, *, split_by: str, week_start: str = "monday") -> None:
    splitter = TimeSeriesSplitByDate(
        n_splits=4,
        date_col=date_col,
        split_by=split_by,
        week_start=week_start,
        gap=1,
        max_train_size=18,
        test_size=5,
    )

    title = f"split_by={split_by!r}"
    if split_by == "weeks":
        title += f", week_start={week_start!r}"
    print(f"\nGenerated folds ({title}):")

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(data), start=1):
        train_max = data.loc[train_idx, date_col].max()
        test_min = data.loc[test_idx, date_col].min()

        print(
            f"Fold {fold_id}: train={len(train_idx)} rows, test={len(test_idx)} rows, "
            f"latest_train={train_max}, earliest_test={test_min}"
        )


def main() -> None:
    date_col = "event_timestamp"
    data = make_synthetic_time_series_dataset(n_samples=32, seed=7, shuffle=True, date_col=date_col)

    print("Rows are shuffled (not chronological):", not data[date_col].is_monotonic_increasing)
    print("First 5 rows:\n", data.head(), sep="")

    _print_folds(data, date_col, split_by="days")
    _print_folds(data, date_col, split_by="weeks")
    _print_folds(data, date_col, split_by="weeks", week_start="sunday")
    _print_folds(data, date_col, split_by="months")


if __name__ == "__main__":
    main()
