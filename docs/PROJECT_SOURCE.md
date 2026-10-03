# Soccer-Prediction-Research — Project Source

## Verified state 2026-10-03
Current GitHub main is authoritative and is re-checked each run; this section intentionally avoids embedding a self-invalidating commit SHA. Latest observed work hardens OOS PIT lineage boolean/cutoff validation, makes research preflight state strictly fail-closed, strengthens research artifact persistence against bounded push/rebase races, and requires meaningful multi-block evidence before predictability research can become a promotion candidate.
README documents a Data Acquisition → Coverage → QC/PIT → Features → Candidate Models → Walk-forward OOS → Metrics/Calibration → Weakness Research → Candidate Validation → Adoption Gate → Registry → Production Prediction architecture. Current prediction is fail-closed when a valid adopted model/current snapshot is unavailable.

## Target isolation
Maintain independent semantics and experience for 1X2, Score Top1/Top3, O/U, BTTS and MOM Top1/Top4. Do not mix incompatible outcomes or retroactively change target definitions.

## Freshness/PIT
retrieved_at_utc is never historical availability evidence. Historical publication evidence is required before PIT verification. Current production requires a successful fresh matchday snapshot; stale prior artifacts are not current prediction evidence.

## Research
Features may use team strength/form, schedule/rest, venue, injuries/lineups, market context, event timing and source-quality metadata when PIT-safe. Rolling windows end at cutoff. OOS is chronological and target-specific calibration must be retained.

## Failure frontier
Prioritize stale-snapshot failures, late lineup information, high-confidence misses, unexplained ranking failures, source conflicts, OOD and competition distribution shift.

## Cross-project governance alignment — 2026-10-03

The five-repository research set is:
- Baseball-Prediction-System
- BTC-Prediction-Research
- 7-Sport-Prediction-Research
- Soccer-Prediction-Research
- Stock-Daily-Prediction-3000

Cross-project transfer is mechanism-level only: DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → ADAPT → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → LOCAL_FROZEN_HOLDOUT → SHADOW → PROMOTE.


A green workflow, artifact existence, model-file existence or external performance claim is not performance verification. Failures/cancellations/skips remain failures/cancellations/skips unless independently rerun and verified. Historical results and holdouts are not rewritten. Adoption is gated on the same chronological OOS baseline by at least 3% relative LogLoss improvement, at least 1% relative improvement in an auxiliary metric, no calibration/accuracy regression, and no metric regression in at least 70% of development evaluation blocks. Cost-unknown, billing-risk or paid-only sources remain HOLD/UNCONFIRMED.

## Runtime research safety

The research runner requires explicit TESTS_PASSED and AUDIT_PASSED boolean handoff values. Missing or invalid values are BLOCKED and never default to pass. Engine retries are transient-only; deterministic schema/logic failures fail fast. Retry history is persisted for diagnosis.
## Recent PIT defense-in-depth hardening — 2026-10-03
`src/research/predictability_calibration.py` now revalidates optional provenance timestamps when supplied by the ledger: `available_at_utc` / `source_available_at_utc`, `published_at_utc`, and `retrieved_at_utc` must remain at or before `prediction_pit_cutoff_utc`; retrieval cannot precede explicit source availability, and retrieval cannot precede publication when both are present. This is defense-in-depth against upstream PIT handoff regressions and remains research-only.

A dedicated regression module `tests/test_predictability_calibration_pit_provenance.py` covers post-cutoff source availability, post-cutoff retrieval, and impossible retrieval-before-source ordering. The research workflow now includes this regression module in its explicit test-file allowlist; current-main Actions must still pass before it is marked VERIFIED.

## Rich matchday evidence layer — 2026-10-04

The existing compact matchday snapshot remains the production input contract. In parallel, `src/data/matchday_rich_enrichment.py` builds a research-only rich evidence surface without changing production model state.

The rich surface can persist, when the registered public/keyless endpoints provide the fields, event status/season/round/week/neutral-site metadata, referee/broadcast/news metadata, ESPN form/rank/record context, multi-provider market dispersion/implied probabilities/overround, recent team results and schedule congestion, SofaScore lineup/manager/pregame-form/H2H/recent cross-competition context, and detailed Open-Meteo weather variables.

Raw provider payloads are retained in `artifacts/matchday_rich_raw.jsonl` with retrieval timestamp and canonical payload hash. Structured fields are retained in `artifacts/matchday_rich_enrichment.csv`. Retrieval time remains distinct from source publication/availability time. The rich layer therefore records `rich_pit_status=UNVERIFIABLE` for historical replay unless independent publication/availability evidence exists.

This layer is acquisition/research evidence, not automatic feature adoption. Candidate use in historical OOS requires the existing PIT → chronological OOS/WFO → calibration → robustness → frozen holdout → adoption gate.


## Rich-source identity hardening — 2026-10-04
FotMob discovery now keys supported domestic competitions by provider league ID and checks the provider country code when available. Display names such as `Premier League` or `Ligue 1` are not sufficient identifiers because the provider can expose same-name competitions in different countries. Unrecognized or mismatched league identities fail closed rather than being silently mapped.

FotMob match-detail retrieval also normalizes numeric IDs that CSV parsing may render as `123.0`, rejects provider `error=true` responses, and verifies the returned `general.matchId` against the requested fixture. Raw error responses remain preserved for forensic lineage but are excluded from valid-payload coverage and enrichment success counts.

## Venue geography integrity — 2026-10-04
A live artifact exposed a provider-level geography defect for Borussia Dortmund: Signal Iduna Park was paired with Aue coordinates. Current BVB information identifies the 2026-10-09 Bundesliga match against Werder Bremen at SIGNAL IDUNA PARK in Dortmund, and a public ESPN API discussion independently records the same Aue mislabel. The research layer therefore uses an explicit provenance-tracked correction for ESPN ger.1 / team 124 / Signal Iduna Park, preserves the provider values, and leaves unknown venue mismatches UNVERIFIED. This correction does not modify production probabilities or the frozen holdout.
