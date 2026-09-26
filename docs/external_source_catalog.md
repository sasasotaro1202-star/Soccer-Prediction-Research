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
| Reep Football Identity Register | entity_data | Stable cross-provider IDs and entity resolution for players, teams, coaches, competitions, seasons and matches | ~1.98M entities / ~7.53M provider bridges / 57 providers in the 2026-09-15 release | Free CC0 download; reference identity layer, not a match-time feature |
| Football-Data.co.uk | fixture_data | Free historical results, match statistics, opening/closing bookmaker odds; current fixture files have documented collection windows | Multiple European + worldwide league datasets; country coverage varies | PIT useful, but current site prohibits automated bots/scrapers/AI use without permission; manual reference only |
| OpenLigaDB | fixture_data | Free fixtures/results and league-season lookup | Bundesliga + many additional leagues; no key; 60 requests/min/IP | Record update/retrieval time for live use |
| API-Football Free Tier | fixture_data | Structured fixtures/events/lineups/statistics/odds/predictions/injuries/transfers | Broad competition endpoint coverage; $0 tier is 100 requests/day and has recent-season limits | Record provider/request timestamps; free tier only |
| TheSportsDB v1 | fixture_data | Fixtures/events/lineups/players/season data + entity fallback | Broad multi-sport database; football coverage varies | Verify provider timestamps and archival availability |
| RSSSF | fixture_data | Historical/rare-competition results and scope discovery | Worldwide country/competition archive, including low-profile leagues | Archive-only; not a match-time feature source |
| World Football Elo Ratings | rating | Independent national-team Elo strength | Worldwide national teams; long history | Date/match-indexed rating; map carefully to prediction time |
| Wyscout Pappalardo Open Dataset | performance_data | Public event-data benchmark | ~1,941 matches from 2017/18 Big Five, 2018 World Cup and Euro 2016 | Historical event data; reconstruct feature availability |
| Impect Open Data | performance_data | Packing, packing-xG and possession-value pxT | Bundesliga 2023/24 sample | Historical sample; verify release timing |
| SkillCorner Open Data | tracking_data | Broadcast tracking, dynamic events and physical aggregates | 10 Australian A-League 2024/25 matches + season aggregates | Research sample, not broad live coverage |
| Metrica Sports Sample | tracking_data | Synchronised tracking and event data | 3 anonymised matches | Historical benchmark only |
| SoccerMon | tracking_data | GPS position, athlete wellness and load reports | Norwegian women's Toppserien, two seasons | Historical sample; not match-time publication |
| Fjelstul World Cup Database | archive | Men's/women's World Cup matches, goals, bookings and squads | Deep tournament archive | Archive source; match-time features need separate verification |
| English Women's Football Database | archive | Women's WSL/Championship matches, appearances and standings | WSL from 2011; Championship from 2014 | Archive source |
| BrazilianFootball/Data | archive | Brazilian Serie A-D and Copa do Brasil results from official CBF dockets | ~2013-2025 | Archive source; verify publication times |
| American Soccer Analysis | performance_data | xG, goals added/g+, xPass and advanced public metrics | MLS, NWSL and USL | Verify provider update semantics |
| FPL-ID-Map | fantasy_data | Crosswalk between FPL and multiple provider IDs | FPL historical ecosystem | Identity only; not a match-time feature |
| Wikidata Football Entity Graph | entity_data | Stable global entity metadata and identifiers | Worldwide public graph | Temporal facts require date filtering |
| SoccerNet | tooling | Video/game-state research benchmark | 550+ broadcast games, ~13 tasks | Raw broadcast video has access restrictions |
| schochastics Football Data | archive | Large historical domestic/international result archive | 1.237M+ matches, 207 domestic leagues + 20 international tournaments, 1888-2023 | Archive source |
| International Results CC0 | archive | Very long national-team result/goalscorer history | 49,000+ matches, 1872-2024 | Archive source |
| Fjelstul English Football Database | archive | Deep English Premier League/EFL history | 208,028 matches, 1888-2024 | Archive source |
| Football.CSV World | fixture_data | Public-domain country/world fixture/result repositories | Brazil, Mexico and broader Football.CSV ecosystem | Match date only; feature availability separate |
| FIFA/Coca-Cola World Ranking | rating | Official national-team strength context | Men's and women's national teams | Update times are coarse; do not infer exact match-time availability |
| BetBrain | forecast | Free public AI/mathematical prediction benchmark using H2H, form, xG and team news | Broad coverage; historical archive availability varies | Historical timestamp must be documented |
| Soccervista | forecast | Free prediction percentages and 1X2-style tips | Many countries and lower-profile leagues | Historical timestamp must be documented |
| FootyStats | performance_data | xG, BTTS, goals, corners, cards, correct scores, form | 1,500+ leagues according to current site | Verify publication/feature availability |
| Soccerway | performance_data | Fixtures, results, lineups, xG, possession, player ratings, H2H | Hundreds of leagues/competitions worldwide | Verify publication time |
| Global Sports Archive | performance_data | Results, fixtures, tables, match statistics, odds movement, transfers | 500+ worldwide soccer leagues/cups/tournaments | Verify provider/retrieval times |
| worldfootball.net | fixture_data | Fixtures, results, standings, historical records and player statistics | Broad competition and historical coverage | Verify publication time |

## Candidate policy notes

- Football-Data.co.uk is useful for PIT methodology and manual reference, but its current terms explicitly restrict automated bots/scrapers/AI use; it must not enter GitHub Actions automation without permission.
- OpenLigaDB is a lightweight no-auth fallback for German and additional league coverage.
- API-Football is only a bounded free-tier candidate: 100 requests/day, no credit card on the free version, and no automatic paid escalation.
- TheSportsDB is a low-volume fallback because free API limits and coverage vary.
- RSSSF is valuable primarily for discovery and long-tail historical coverage rather than live prediction features.
- BetBrain and Soccervista are external forecast benchmarks only; their published percentages should be tested, not trusted.
- FootyStats, Soccerway and Global Sports Archive are broad enrichment/scope sources; use them to expand cases and feature channels before considering fusion.

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
- Open-data model research: Wyscout/Impect/SkillCorner/Metrica/SoccerMon provide small but high-information event/tracking samples for testing representations, tracking-derived features and predictability ceilings.
- Historical breadth: schochastics, International Results, Fjelstul and Football.CSV add long-run and lower-tier coverage without treating archival results as prediction-time features.
- Identity spine: Reep should be tested as the canonical cross-provider join layer before building many one-off name/ID mappings.
- Router research: test whether external-source disagreement predicts internal-model error, without assuming the external source is superior.

Every lane remains research/shadow until it passes chronological OOS, calibration, robustness, PIT/replay and promotion gates.