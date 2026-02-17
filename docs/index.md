# TimeSeriesSplitByDate

`TimeSeriesSplitByDate` is a scikit-learn-compatible cross-validator for machine learning with irregular timestamps.

It is designed for datasets where index-based splits can misrepresent real time windows because observations are unevenly spaced.

Key capabilities:

- Date-based splitting modes: `days`, `weeks`, and `months`
- Configurable week start (`monday` or `sunday`) for weekly folds
- Deterministic chronological train/test boundaries from datetime values
- Fallback to sklearn `TimeSeriesSplit` when no date source is provided

Use this package when you need validation folds that reflect calendar time, not just row counts.
