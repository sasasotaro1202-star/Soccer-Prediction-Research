# Broad External Soccer Intelligence Sources (Research Only)

This catalog is deliberately wider than external match forecasts. Sources are separated by the information type they can contribute to the prediction stack.

## Current source candidates

| Source | Class | Contribution | Breadth / constraint | PIT handling |
| --- | --- | --- | --- | --- |
| Forebet | forecast | 1X2 probabilities, correct score, goals, BTTS, handicap, cards, corners | Very broad global coverage; current site/FAQ describe 850+ to 1,200+ leagues | Historical availability timestamp required |
| PredictZ | forecast | 1X2 odds display, correct-score tips, recent form | Broad global coverage | Historical publication timestamp generally UNKNOWN |
| Opta Analyst | forecast | External 1X2 probabilities / season forecasts | Major competitions; public history incomplete | Timestamp must be documented per row |
| Oddspedia SmartBet | forecast | 1X2 model probabilities; injury/player/H2H context; odds comparison | Broad but competition coverage varies | Historical timestamp must be documented |
| ClubElo | rating | Date-specific team strength and home-field adjustment | Long historical European club coverage | Use date-aligned rating snapshot |
| FiveThirtyEight SPI | forecast | Historical pre-match SPI + forecast benchmark | Published match files go back to 2016 | Date-level archive; strict PIT needs timestamped availability |
| StatsBomb Open Data | performance_data | Events, lineups, selected 360 data | Selected competitions/seasons | Verify feature availability at prediction time |
| Understat | performance_data | Shot-level xG / shot locations | Big 5 + RFPL in supported public ecosystem | Treat as delayed; verify availability |
| FBref | performance_data | Team/player advanced statistics | Wide coverage; metric availability varies | Verify publication/availability time |
| Transfermarkt | player_status | Injuries, suspensions, transfers, squad context | Very broad player/club coverage | Verify effective status + publication time |
| Open-Meteo | weather | Historical weather + archived forecast runs | Global | Archived forecast run timestamp is preferred |
| OpenFootball World | fixture_data | Fixture/result coverage and competition discovery | North America, Asia, Africa, Australia, Europe and more | Match date is not a feature-availability timestamp |
| OpenFootball Players | entity_data | Player/entity normalization and discovery | Country-by-country reference data | Reference only; not a match-time feature |

## Operating rules

1. External forecasts are priors/benchmarks, not truth.
2. Unknown historical availability stays FAIL-CLOSED.
3. Forecast, rating, performance, player-status and weather channels are evaluated separately before any fusion.
4. Source disagreement, incremental information, error correlation, calibration and regime-specific value are measured before changing production predictions.
5. Catalog membership never implies production approval.
6. The catalog is a discovery frontier and should keep expanding when new free, broad and reproducible sources appear.

## Immediate research lanes

- Forecast fusion: internal + Opta + Forebet + PredictZ + Oddspedia.
- Historical benchmark: FiveThirtyEight SPI + ClubElo.
- Process features: Understat + FBref + StatsBomb.
- Availability shocks: Transfermarkt + official competition/club announcements where timestamps are recoverable.
- Environment: Open-Meteo historical forecast runs at multiple lead times.
- Scope expansion: OpenFootball World + OpenFootball Players for competition/entity discovery.
- Router research: test whether external-source disagreement predicts internal-model error, without assuming the external source is superior.

Every lane remains research/shadow until it passes chronological OOS, calibration, robustness, PIT/replay and promotion gates.