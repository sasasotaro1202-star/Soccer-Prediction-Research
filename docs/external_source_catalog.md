# Broad External Soccer Intelligence Sources (Research Only)

This catalog is deliberately wider than external match forecasts. Sources are separated by the information type they can contribute to the prediction stack.

## Priority research candidates

| Source | Class | What it can contribute | Breadth / constraint | PIT handling |
| --- | --- | --- | --- | --- |
| Forebet | forecast | 1X2 probabilities, correct score, goals, BTTS, handicap, cards, corners | Very broad global coverage; site/FAQ currently describe 850+ to 1,200+ leagues | Historical rows require documented availability; otherwise UNKNOWN |
| PredictZ | forecast | 1X2 odds display, correct-score tips, recent form | Broad global coverage | Historical publication timestamp generally UNKNOWN |
| Opta Analyst | forecast | External 1X2 probabilities / season forecasts | Major competitions; public history incomplete | Timestamp must be documented per row |
| Oddspedia SmartBet | forecast | 1X2 model probabilities; injuries/player stats/H2H context; odds comparison | Broad but competition coverage varies | Historical timestamp must be documented |
| ClubElo | rating | Date-specific team strength, historical Elo, home-field adjustment | Long historical European club coverage | Rating snapshot itself is date-indexed; map snapshot date to prediction time |
| FiveThirtyEight SPI | forecast | Historical pre-match SPI + forecast benchmark | Published match files go back to 2016; historical/archival use | Date-level history is useful for benchmark; strict PIT requires timestamped availability |
| StatsBomb Open Data | performance_data | Events, lineups, selected 360 data | Selected competitions/seasons | Use event times and verify feature availability |
| Understat | performance_data | Shot-level xG / shot locations | Big 5 + RFPL in the supported public ecosystem | Treat as delayed and verify availability |
| FBref | performance_data | Team/player advanced statistics | Broad but metric/competition coverage varies | Verify publication/availability time |
| Transfermarkt | player_status | Injuries, suspensions, transfers, squad context | Very broad player/club coverage | Effective status time + publication time must be checked |
| Open-Meteo | weather | Historical weather and archived forecast runs at lead times | Global | Prefer archived forecast runs for PIT-safe reconstruction |

## Operating rules

1. External forecasts are priors/benchmarks, not truth.
2. No historical source with unknown availability timestamp is silently converted to PIT-safe data.
3. A source can be useful without being a forecast source: rating, xG, player status and weather should enter as separate information channels.
4. Source redundancy is intentional. Independent sources are evaluated for disagreement, incremental information, error correlation and regime-specific value.
5. Research status remains separate from production status. No source is promoted from this catalog by catalog membership alone.
6. The catalog should be expanded whenever a new broad-coverage, free and reproducible source is discovered.

## Immediate experiments

- Forecast fusion: Opta + Forebet + PredictZ + Oddspedia + internal model.
- Historical benchmark: FiveThirtyEight SPI + ClubElo.
- Process features: Understat + FBref + StatsBomb.
- Availability shock: Transfermarkt + official club/league announcements where timestamps are recoverable.
- Environment: Open-Meteo historical forecast runs by lead time.
- Cross-source routing: estimate whether source disagreement predicts internal-model error before changing the forecast itself.

All experiments must use chronological OOS, calibration, robustness and PIT/replay checks.