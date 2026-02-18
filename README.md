# TimeSeriesSplitByDate

[![PyPI version](https://img.shields.io/pypi/v/timeseriessplitbydate.svg)](https://pypi.org/project/timeseriessplitbydate/)
[![CI](https://github.com/theyou21/timeseriessplitbydate/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/theyou21/timeseriessplitbydate/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/timeseriessplitbydate.svg)](https://pypi.org/project/timeseriessplitbydate/)
[![License](https://img.shields.io/pypi/l/timeseriessplitbydate.svg)](https://pypi.org/project/timeseriessplitbydate/)


Date-aware cross-validation for machine learning on irregularly timestamped data.

## Installation

```bash
pip install timeseriessplitbydate
```

## Motivation / Problem Statement

Many ML pipelines split by row index, not by actual event time. That can be misleading when timestamps are irregular (bursts, long gaps, uneven logging), because a fold with 500 rows may represent 2 days while another fold with 500 rows may represent 3 weeks.

`sklearn.model_selection.TimeSeriesSplit` is excellent for many cases, but its documentation explicitly assumes equally spaced samples for comparable fold durations. See: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>

`TimeSeriesSplitByDate` addresses this by splitting on real dates (`days`, `weeks`, `months`) while preserving sklearn-style usage.

## When should I use this?

- Your observations are time-indexed but not equally spaced.
- You need validation windows aligned to calendar units (day/week/month).
- You want a strict chronological split with explicit date boundaries.
- You still want an sklearn-compatible CV object for model selection workflows.
- You want a fallback to `TimeSeriesSplit` when date data is not provided.

## Comparison vs sklearn TimeSeriesSplit

| Capability | `TimeSeriesSplit` | `TimeSeriesSplitByDate` |
| --- | --- | --- |
| Split basis | Sample index order | Datetime boundaries (`days`, `weeks`, `months`) |
| Assumption for comparable fold duration | Equally spaced samples | No equal-spacing assumption |
| Handles irregular timestamps directly | Limited | Yes |
| Calendar-aligned weekly/monthly folds | No | Yes |
| Week start control | No | Yes (`week_start='monday'` or `'sunday'`) |
| Gap meaning in date-based mode | Samples | Calendar days |
| sklearn-style API (`split`, `get_n_splits`) | Yes | Yes |

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
    split_by="weeks",
    week_start="monday",
)

for train_idx, test_idx in splitter.split(X):
    print(train_idx, test_idx)
```

## Realistic ML Examples

### 1) Healthcare: irregular patient encounters

Patients rarely generate observations at fixed intervals. Encounters cluster around acute events and then become sparse. Date-based folds prevent evaluation leakage where training accidentally includes data too close in time to validation encounters.

### 2) Finance: uneven market/event cadence

Signals around earnings, macro releases, or volatility spikes are not uniformly spaced in event time. Date-based weekly/monthly folds let you evaluate stability across comparable calendar periods rather than equal row counts.

## FAQ / Gotchas

### Do I need to sort rows by date first?

No. Splits are built from datetime values, not row order. Chronological boundaries are date-based even if your DataFrame is shuffled.

### What happens with duplicate timestamps?

Duplicates are handled deterministically by date masks. Multiple rows sharing the same timestamp are kept together according to the fold boundaries for that mode.

### Are timezone-aware datetimes supported?

Yes, as long as your datetime column is consistently parseable by pandas. Avoid mixing incompatible timezone formats within the same column.

### What exactly does `gap` mean in date-based modes?

- `split_by='days'`: `gap` is calendar days between train and test windows.
- `split_by='weeks'` / `split_by='months'`: `gap` is calendar days added to the start of the test period.
- Fallback `TimeSeriesSplit`: `gap` is in number of samples (sklearn behavior).

## Documentation and Demo

- Usage guide: <https://github.com/theyou21/timeseriessplitbydate/blob/main/docs/usage.md>
- Runnable example: <https://github.com/theyou21/timeseriessplitbydate/blob/main/examples/basic_usage.py>
- Notebook demo: <https://github.com/theyou21/timeseriessplitbydate/blob/main/notebooks/demo.ipynb>

## Citing

If you use this in academic work, please cite:

```bibtex
@software{timeseriessplitbydate_2026,
  author  = {Bai, Zeyu},
  title   = {TimeSeriesSplitByDate},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/theyou21/timeseriessplitbydate}
}
```

## Development

```bash
pip install -e .[dev]
ruff check .
pytest
```
