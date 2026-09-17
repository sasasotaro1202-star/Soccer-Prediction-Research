# External football data source integration matrix

This matrix defines the integration order for the eight requested sources. It is an engineering registry, not a claim that every source is suitable for production.

## Adoption rule

A source is **not production-eligible merely because it can be downloaded**. The source must pass:

1. Coverage check by competition/season/field.
2. Exact source timestamp / publication-time reconstruction where required.
3. Point-in-time (PIT) validation: `feature_available_at <= prediction_cutoff_at`.
4. Leakage and current-match contamination checks.
5. Schema/type/identity/reconciliation checks.
6. Chronological walk-forward OOS evaluation on the same windows as the incumbent.
7. Stability and calibration checks.
8. Reproducible cache/content digest checks.

Standard prediction cutoff remains T-60 minutes where the prediction workflow uses a fixed cutoff. Confirmed lineups after that cutoff must not be used for historical predictions.

## Source matrix

| Priority | Source | Acquisition | API key | Free status | PIT feasibility | Initial role | Production now |
|---:|---|---|---|---|---|---|---|
| 1 | ClubElo | dated public HTTP API/history | No | Public/free | High, subject to date-alignment audit | team-strength prior / Elo supplement | No |
| 2 | StatsBomb Open Data | public GitHub JSON | No | Free/open research data | Medium-high where historical event date is sufficient | event/xG feature research | No |
| 3 | Understat | public web extraction | No | Public web access | Medium; publication-time reconstruction must be demonstrated | xG/shot supplement | No |
| 4 | Open-Meteo | public weather API | No | Free API | Medium; use only forecast information known at cutoff | weather feature experiment | No |
| 5 | SofaScore | public web/API-style endpoints | No known key required for public access | Public access; verify usage constraints | Low-medium until timestamp replay is demonstrated | lineups/events/odds supplement | No |
| 6 | API-Football | REST API | Yes (`API_FOOTBALL_KEY`) | Free tier exists; quota/history must be verified | Medium-high if archived snapshots or timestamps are available | structured fixtures, lineups, injuries, odds | No |
| 7 | FBref | public web pages | No | Public web access | Medium; historical/PIT availability must be demonstrated | team/player statistics supplement | No |
| 8 | Sportmonks | REST API | Yes (`SPORTMONKS_API_TOKEN`) | Free plan exists; coverage/quota must be verified | Medium-high if timestamped historical data is available | structured fallback/supplement | No |

## Implementation order

### Phase A: low-cost, high-control sources

- ClubElo
- StatsBomb Open Data
- Understat
- Open-Meteo

These are registered first and tested independently. They must not alter production predictions until the adoption gate passes.

### Phase B: structured/API supplements

- SofaScore
- API-Football
- FBref
- Sportmonks

These are added only where Phase A or the incumbent sources leave a verified coverage gap or a demonstrable incremental-information opportunity.

## Connect role split

- **GitHub:** source code, adapters, tests, cache, artifacts, Actions and fail-closed gates.
- **Football app:** current competition/team/match context and interactive verification.
- **Tavily:** source discovery, current documentation/pricing/coverage checks, and cross-source research.
- **Parallel Search:** fast broad discovery and corroboration.
- **Firecrawl:** targeted page extraction when a web source must be inspected structurally.
- **Context7:** current library/API implementation documentation.
- **Data Analytics:** data-quality profiling, chronological OOS, calibration and adoption analysis.
- **Wolfram:** numerical/statistical verification when specialized calculations are needed.

## Reliability and cache rules

- Raw responses are cached by source, request identity, and content digest.
- A cache hit must preserve the original source timestamp metadata.
- A stale intermediate artifact must not be silently reused after an upstream data/schema change.
- Transient HTTP failures use bounded retry/backoff; permanent schema/PIT failures fail closed.
- Missing/unavailable data is never converted to a numeric zero.
- Production eligibility is an explicit state transition, never an implicit default.
