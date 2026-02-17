# FAQ

## Do I need to sort rows by date first?

No. Splits are computed from datetime values and date boundaries, not row order.

## What happens with duplicate timestamps?

Duplicate dates are handled deterministically using date masks. Rows with the same timestamp are assigned according to fold boundaries.

## Are timezone-aware datetimes supported?

Yes, as long as the datetime values are consistently parseable by pandas.

## What does `gap` mean in date-based modes?

- `split_by="days"`: `gap` is calendar days between train and test windows.
- `split_by="weeks"` / `split_by="months"`: `gap` is calendar days added to the test-window start.
- Fallback sklearn `TimeSeriesSplit`: `gap` is in number of samples.
