# TimeSeriesSplitByDate

Date-aware splitting utility for time series model evaluation.

## Installation

```bash
pip install timeseriessplitbydate
```

For local development:

```bash
pip install -e .
```

## Quickstart

```python
import pandas as pd

from timeseriessplitbydate import TimeSeriesSplitByDate

X = pd.DataFrame(
    {
        "feature": range(8),
        "event_date": pd.date_range("2024-01-01", periods=8, freq="D"),
    }
)

splitter = TimeSeriesSplitByDate(
    n_splits=3,
    date_col="event_date",
    split_by="days",
)

for train_idx, test_idx in splitter.split(X):
    print(train_idx, test_idx)
```

## How it differs from sklearn TimeSeriesSplit

- Supports date-column-driven splitting (`date_col`) for DataFrame inputs.
- Adds calendar-aware modes:
  - `split_by="days"`: equal-width date-range intervals.
  - `split_by="weeks"`: adjacent week folds with `week_start="monday"` (default) or `week_start="sunday"`.
  - `split_by="months"`: adjacent month folds.
- Supports external date data via `set_date_data(...)`.
- Falls back to sklearn `TimeSeriesSplit` when no date source is provided.
- In date-based modes, `gap` is interpreted as calendar days.

## Development

```bash
pip install -e .[dev]
ruff check .
pytest
```

## More Documentation

- Usage guide: `docs/usage.md`
- Runnable demo: `examples/basic_usage.py`
