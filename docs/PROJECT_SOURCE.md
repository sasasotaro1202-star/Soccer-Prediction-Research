# Soccer-Prediction-Research — Project Source

## Verified state 2026-10-03
main latest observed HEAD after re-check: 0643e13d39dd73bb783044c411a0083a5dbd95c9. Latest observed work after the documentation commit hardens OOS PIT lineage boolean/cutoff validation and adds strict OOS PIT guard tests.
README documents a Data Acquisition → Coverage → QC/PIT → Features → Candidate Models → Walk-forward OOS → Metrics/Calibration → Weakness Research → Candidate Validation → Adoption Gate → Registry → Production Prediction architecture. Current prediction is fail-closed when a valid adopted model/current snapshot is unavailable.

## Target isolation
Maintain independent semantics and experience for 1X2, Score Top1/Top3, O/U, BTTS and MOM Top1/Top4. Do not mix incompatible outcomes or retroactively change target definitions.

## Freshness/PIT
retrieved_at_utc is never historical availability evidence. Historical publication evidence is required before PIT verification. Current production requires a successful fresh matchday snapshot; stale prior artifacts are not current prediction evidence.

## Research
Features may use team strength/form, schedule/rest, venue, injuries/lineups, market context, event timing and source-quality metadata when PIT-safe. Rolling windows end at cutoff. OOS is chronological and target-specific calibration must be retained.

## Failure frontier
Prioritize stale-snapshot failures, late lineup information, high-confidence misses, unexplained ranking failures, source conflicts, OOD and competition distribution shift.
