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