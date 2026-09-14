# Competition production gate

A competition may be researched broadly but may enter production only when every required gate is explicitly true.

| Gate | Requirement |
|---|---|
| Source | Identifiable, reproducible source and provenance |
| Availability | `source_available_at_utc` is auditable and is not substituted with retrieval time |
| PIT | Pre-kickoff replay verification meets the configured minimum |
| Quality | Parsing, duplicates, identity mapping, completeness and consistency checks pass |
| Features | All production features are available without post-kickoff information |
| OOS | Chronological walk-forward evaluation is valid |
| Calibration | Probability calibration is evaluated on unseen chronological data |
| Sample | Minimum effective sample size is met |
| Adoption | Explicit adoption gate and registry state are `ADOPT` |

Failure of any gate means `production_eligible=false` and the competition is excluded or abstained from in production. No fallback may silently bypass a failed gate.

The initial target set includes Premier League, Championship, Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, UCL, UEL, J1/J2/J3, DFB-Pokal, Friendly, EFL-related competitions, plus major domestic cups and other major competitions where reliable auditable data can be established.

Domestic cup examples include FA Cup, Carabao Cup/EFL Cup, Copa del Rey, Coppa Italia, Coupe de France, KNVB Cup, Taça de Portugal, Belgian Cup, Scottish Cup, Turkish Cup, Greek Cup, J.League Cup and Emperor's Cup. These names are research targets, not automatic production approvals.
