# Rich Matchday Data Layer

The matchday acquisition path now keeps a research-only rich evidence layer in addition to the compact production snapshot.

## Coverage

The rich layer can persist, when supplied by the existing public/keyless or already-registered endpoints:

- event status, season, round/week, neutral-site metadata and venue
- referee and broadcast metadata
- ESPN pre-match form/rank/record metadata
- multi-provider market observations, implied probabilities, provider dispersion and overround
- recent team results, goals, points, goal difference, BTTS/Over-2.5 rates, recent opponents and schedule-congestion counts
- SofaScore lineup formations, starter/substitute IDs, captain IDs and missing-player reasons
- SofaScore managers, pregame-form payloads, H2H history and cross-competition recent matches
- Open-Meteo temperature, humidity, dew point, apparent temperature, precipitation/rain/showers/snowfall, pressure, visibility, cloud layers, wind speed/direction/gusts, daylight, sunshine duration, CAPE and weather code
- raw provider JSON records with retrieval timestamp and payload hash

Open-Meteo documents these weather-variable families in its forecast APIs. See the current provider documentation: open-meteo.com/en/docs. Sofascore endpoint references used here include event detail, lineups, managers, pregame-form, H2H and team last-events endpoints.

## PIT

retrieved_at_utc is retained separately from source publication/availability. Current collection therefore records the rich layer as UNVERIFIABLE for historical PIT unless an independent archive/availability proof exists.

The raw JSONL is evidence for replay/debugging, not automatic permission to use the values in historical OOS.

## Integration

artifacts/matchday_rich_enrichment.csv is the structured research surface. artifacts/matchday_rich_raw.jsonl is the raw lineage surface. Production predictions continue to use the existing compact fixture path until a target-specific chronological OOS/robustness/holdout gate proves incremental value.

The rich layer does not change production_changed, model registry state, or target semantics.
