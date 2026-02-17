# Usage Guide

`TimeSeriesSplitByDate` is a cross-validator for date-aware model validation.
It keeps training data earlier than validation data and supports the same core knobs as `TimeSeriesSplit` (`n_splits`, `max_train_size`, `test_size`, `gap`).

This guide uses fully synthetic data only.

## Quick Demo (Uneven + Shuffled + Date Column)

```python
from timeseriessplitbydate import TimeSeriesSplitByDate, make_synthetic_time_series_dataset

DATE_COL = "event_timestamp"
X = make_synthetic_time_series_dataset(n_samples=32, seed=7, shuffle=True, date_col=DATE_COL)

splitter = TimeSeriesSplitByDate(
    n_splits=4,
    date_col=DATE_COL,
    split_by="weeks",
    week_start="monday",  # default
    gap=1,
    max_train_size=18,
    test_size=5,
)

for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    train_end = X.loc[train_idx, DATE_COL].max()
    test_start = X.loc[test_idx, DATE_COL].min()
    print(f"Fold {fold_id}: train_end={train_end}, test_start={test_start}")
```

This works even when rows are shuffled, because splitting is driven by the datetime values in `date_col`, not row order.

## Date Input Sources

The splitter looks for date information in this order:

1. `splitter.set_date_data(<iterable>)`
2. `date_col` in a pandas DataFrame

## Split Modes

- `split_by="days"`: divides the full observed date span into equal contiguous time intervals and builds expanding train windows.
- `split_by="weeks"`: uses adjacent calendar weeks; set `week_start="monday"` (default) or `week_start="sunday"`.
- `split_by="months"`: uses adjacent calendar months (train through month *i*, test on month *i+1*).

For `split_by="weeks"` and `split_by="months"`, the effective number of folds is determined by the number of unique periods in the input date data (`periods - 1`).

## Fallback Behavior

If no date source is available (`set_date_data` or `date_col`), the class falls back to scikit-learn's `TimeSeriesSplit`.

## Example Script

Run the end-to-end example from project root:

```bash
PYTHONPATH=src python examples/basic_usage.py
```
