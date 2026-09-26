# External Forecast Intelligence (Research Only)

This layer treats third-party forecasts as **external probability priors and benchmarks**, not as production truth.

## Design

- **PIT fail-closed:** `source_available_at_utc` must be known and no later than `prediction_time_utc`.
- **Provenance required:** every forecast row carries a source identifier and replayable URL.
- **No automatic promotion:** the fusion weight is a research parameter learned only on a chronological training slice.
- **Disagreement is first-class:** L1 probability distance, top-class disagreement, and confidence deltas are emitted for downstream routing/error analysis.
- **Production isolation:** this module never changes the Champion model and should remain research/shadow until full OOS, calibration, robustness, replay and promotion gates pass.

## Supported source classes

### Opta Analyst
Opta publishes match-outcome probabilities using its Opta Power Rankings together with betting-market odds. Its public prediction hub provides individual game predictions and season projections. Public historical coverage is incomplete, so only forecasts with a documented prediction timestamp and source availability should enter the research table.

### Market priors
Football-Data publishes free historical results, match statistics, and bookmaker odds. The site documents that opening and closing odds are collected at defined fixture-list times, but exact row-level publication timestamps are not present in the historical CSVs. Therefore this project must **not infer exact PIT from retrieval time**; use only independently documented availability bounds or leave the row unverified.

### Understat xG
Understat exposes shot-level xG for major European leagues. It is useful as a historical enrichment source, but reported data can lag after matches, so it should be treated as a delayed feature source and validated per prediction time.

## Contract example

```text
match_id,source,prediction_time_utc,source_available_at_utc,p_home,p_draw,p_away,provenance_url
m123,optA,2026-09-20T12:00:00Z,2026-09-20T09:00:00Z,0.52,0.27,0.21,https://…
```

The fusion rule is the convex linear pool:

`P_fused = (1-w) P_internal + w P_external`

The weight `w` must be fit strictly inside the chronological training slice. Do not tune it on latest-unseen or frozen/blind holdout data.
