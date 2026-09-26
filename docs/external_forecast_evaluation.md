# External Forecast Evaluation

This module reuses the project's standard 1X2 metrics and applies them to
PIT-verified external forecasts.

Reported metrics include:

- Accuracy
- LogLoss
- Brier
- ECE
- class support / recall / precision

For internal-vs-external comparison, the exact same validated rows are scored for
both forecasts. A linear-pool blend is evaluated only when its weight was chosen
on an earlier chronological training slice.

No result from the latest-unseen or frozen/blind holdout may be used to select the
blend weight.

## Interpretation

An external source is not promoted merely because it has better Accuracy.
LogLoss/Brier/calibration, chronological stability, important-case performance,
and PIT/provenance integrity must also be evaluated.
