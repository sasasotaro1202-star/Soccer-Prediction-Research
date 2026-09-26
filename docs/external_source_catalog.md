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
| Global Football (Soccer) Data Lake | archive | Quality-gated global fixtures, lineups, xG, odds and league catalogue | 271 leagues, 673,966 fixtures, 11,104 teams, 182,125 players in current release | Use known_at when present; inspect league history_status and CC-BY-4.0 license |
| Dynasty Scouting League 2024 | performance_data | Open event data, lineups and match metadata | 2024 JSONL sample | Reconstruct feature availability |
| wosostats | performance_data | Women's USWNT/NWSL event benchmark | 2016 onward; community-collected and largely dormant | Audit source quality and event timestamps |
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

| K League Official | live_data | Official Korean K League 1/2 fixtures/results and current competition match-centre schedule | Current schedule also exposes Korean Cup and AFC club competition selectors | Official page timestamps/retrieval time required |
| WE League Official Data Site | live_data | Official Japanese women's fixture/results/venue/attendance dataset | Current 2026/27 data covers league and cup competition rows | Official data-site availability time required |
## Regional and high-coverage discovery

| Source | Class | Contribution | Breadth / constraint | PIT handling |
| --- | --- | --- | --- | --- |
| PlaymakerStats | fixture_data | Competition/team/player discovery, fixtures, reports and context | Current site reports 6.22M fixtures, 2.84M players and 5,126 competitions | Verify page publication time and automation terms |
| BDFutbol | fixture_data | Deep Spanish history and lower-tier/women's coverage | Current site reports 377k matches and 175k players | Verify publication time and automation terms |
| J.League Official Match Centre | live_data | Official Japan fixtures, starting-XI status and match data | J.League and supported Japanese cups/competitions | Record JST event/publication/retrieval times |
| Kooora | live_data | Regional live scores, fixtures, standings and football news | Strong Middle East/North Africa + international coverage | Verify timestamps and terms |

| OpenFoot API | live_data | Fixtures, results, live events, confirmed lineups and xG | Public beta; no key required for initial public testing | Preserve response/request timestamps and verify terms/rate limits |
| BSD (Bzzoiro Sports Data) | live_data | Fixtures, live, odds, lineups, injuries, transfers and predictions | Current docs claim 30+ leagues, 3,000+ teams, 8,900+ players and free access | Preserve provider/request timestamps; treat provider forecasts as external benchmark |

## Broad live-provider and collector layer

| Source | Class | Contribution | Key constraint |
| --- | --- | --- | --- |
| Sofascore public web/app data | live_data | Live state, xG, shotmaps, lineups, ratings, H2H, tournaments | No official public API; treat endpoints as unofficial and verify terms/stability |
| FotMob public web/app data | live_data | xG, shots, momentum, lineups, ratings, news, transfers, fixtures | Public app data; verify terms and endpoint stability |
| soccerdata | tooling | Unified Python access to multiple football providers | Provider-specific PIT/terms still apply |
| worldfootballR | tooling | R collection layer for FBref/Understat/transfer/match datasets | Provider-specific PIT/terms still apply |
| Kaggle European Soccer DB | archive | 25k+ historical matches, 10k+ players, events, lineups, multi-provider odds | 2008-2016 archive; reconstruct timing before PIT use |

## Open modelling and discovery tooling

| Tool | Class | Contribution | Breadth / constraint |
| --- | --- | --- | --- |
| Kloppy | tooling | Normalises event/tracking data across provider formats | 16+ provider formats; BSD-3 |
| socceraction | tooling | SPADL + xT/VAEP action-value features | Event-data transformation; source PIT still required |
| PySport Open-Source Index | tooling | Discovery of additional open football/sports tools | Dozens of projects; discovery source only |

## Timestamped and long-running external forecast benchmarks

| Source | Class | Contribution | Current public scope/history | PIT handling |
| --- | --- | --- | --- | --- |
| SoccerPortalX | forecast | Multi-stage pre-kickoff probabilities, live refreshes, model-versioned archive | 153 competitions; 421k+ graded/timestamped forecasts; corpus 1.6M+ matches / 1,243 competitions / 34 years | Strong candidate: verify per-row archive timestamp and terms |
| Foresportia | forecast | 1X2 + BTTS/O-U/DNB + score candidates + stability | 40+ current competitions; public history says 18k+ verified matches in 53 competitions | Use per-row published/update timestamp; keep strict PIT |
| Tofiko | forecast | Poisson/Dixon-Coles/Elo 1X2 probabilities + market comparison | 30+ leagues; 800+ tracked forecasts | Verify exact forecast-publication time |
| ProSoccer.GR | forecast | Neural/computational forecast, score and O/U outputs | 150+ leagues/cups | Weekly/daily update schedule; reconstruct exact row availability |
| The Football Simulator | forecast | Transparent Monte Carlo match/season probabilities | 12 current competitions; 36k+ historical results and 95 CSV files | Verify row timestamp before replay |

## Market-odds and referee channels

| Source | Class | Contribution | Current free availability | PIT handling |
| --- | --- | --- | --- | --- |
| Odds-API.io | odds_market | Pre-match/live bookmaker odds and market snapshots | Free tier: 100 req/hour, 500/day, 2 recreational books | Preserve request/provider timestamps; distinguish snapshots from closing prices |
| PulseScore | odds_market | Multi-book canonical odds and freshness | Free BASIC: 500 requests/month | Preserve provider/request time |
| StatsBet Referees | performance_data | Cards, goals, penalties and referee histories | 130+ leagues | Appointment availability must precede prediction time |
| RefsRadar | performance_data | Referee cards/fouls/penalties and home-bias features | 27 leagues / 16 countries | Appointment time required for pre-match feature |
| ScorelineAI Referees | performance_data | Expected cards and discipline probabilities | 1,600+ active referees | Appointment time required; research only |

## News and live-state channels

| Source | Class | Contribution | Breadth / constraint | PIT handling |
| --- | --- | --- | --- | --- |
| football-data.org | live_data | Structured competitions, matches, standings, teams and players | Current v4 docs list 157 competitions; free access is rate-limited | Record provider/request time |
| ESPN Soccer Scoreboard endpoint | live_data | Low-latency live status and finished results | Competition coverage follows ESPN; endpoint is unofficial | Record response/retrieval time and endpoint version |
| GDELT DOC 2.0 | news | Global multilingual football-news monitoring, search, tone and timelines | Global news corpus; keyless public API | Source publication/indexing time must be separated |
| GDELT GKG | news | Themes, entities, organizations, locations and context | Global multilingual news graph | Ingestion time is not publication time |
| BBC Sport Football RSS | news | Team/league football news feeds | Premier League, EFL, European, women's and team feeds are available | Feed/article timestamps required |

## Venue, travel and environment support

| Source | Class | Contribution | Breadth / constraint | PIT handling |
| --- | --- | --- | --- | --- |
| OpenFootball Clubs & Stadiums | venue_data | Global club/stadium identity and aliases | Worldwide public-domain club/stadium data | Date-filter venue changes |
| WorldSoccerStadiums | venue_data | Stadium coordinates and capacity | 4,887 stadiums in the public dataset | Static reference; verify historical venue changes |

These sources can be transformed into derived features such as home/away venue continuity, travel distance, altitude difference, capacity-normalized attendance (when attendance is separately available), and geographic novelty. Derived features must use only venue information known by prediction time.

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

- Forecast fusion: internal + Opta + Forebet + PredictZ + Oddspedia + SoccerPortalX + Foresportia + Tofiko + ProSoccer + The Football Simulator.
- Forecast archive replay: prioritize SoccerPortalX/Foresportia where historical timestamps and raw probability rows can be independently recovered.
- Historical benchmark: FiveThirtyEight SPI + ClubElo.
- Process features: Understat + FBref + StatsBomb.
- Availability shocks: Transfermarkt + official competition/club announcements where timestamps are recoverable.
- Environment: Open-Meteo historical forecast runs at multiple lead times.
- Scope expansion: OpenFootball World + OpenFootball Players for competition/entity discovery.
- Open-data model research: Wyscout/Impect/SkillCorner/Metrica/SoccerMon provide small but high-information event/tracking samples for testing representations, tracking-derived features and predictability ceilings.
- Historical breadth: schochastics, International Results, Fjelstul and Football.CSV add long-run and lower-tier coverage without treating archival results as prediction-time features.
- Identity spine: Reep should be tested as the canonical cross-provider join layer before building many one-off name/ID mappings.
- Router research: test whether external-source disagreement predicts internal-model error, without assuming the external source is superior.
- Information-shock research: GDELT + BBC feeds can be transformed into time-stamped news shock/state variables and tested for incremental predictive value.
- Representation research: Kloppy + socceraction can turn newly discovered event sources into common SPADL/xT/VAEP representations, reducing source-specific implementation work.
- Broad acquisition research: soccerdata/worldfootballR should be evaluated as collectors, not as a single information source; every downstream provider remains independently PIT-audited.
- Live cross-checking: Sofascore/FotMob can be used as redundant state channels, with disagreements feeding data-quality and source-reliability monitoring.
- Live-state research: football-data.org + ESPN can be used as redundant schedule/status channels, with cross-source disagreement monitored as a data-quality signal.
- Market research: Odds-API.io/PulseScore can provide free bounded odds snapshots for market-prior and price-movement experiments; no automatic paid escalation.
- Referee research: appointment-aware referee features can augment card/penalty/O-U models and potentially provide an external state variable for match volatility.
- Regional coverage research: J.League Official + Kooora + BDFutbol + PlaymakerStats should be used to discover competitions that are missing from the current scope before feature/model work begins.

Every lane remains research/shadow until it passes chronological OOS, calibration, robustness, PIT/replay and promotion gates.
- Global breadth benchmark: the current open Global Football Data Lake release exposes 271 leagues, 673,966 fixtures, 11,104 teams and 182,125 players, with `known_at` leakage metadata; use it for research coverage/QA, not as an automatic production truth source.
