# Soccer-Prediction-Research — Project Source

## Verified state 2026-10-03
audit reference HEAD: 779f5881899780a57578fdd89216a878059a7fdc (2026-10-03T04:26:18Z). This is an audit baseline, not a promise that main remains unchanged after the audit.
README documents a Data Acquisition → Coverage → QC/PIT → Features → Candidate Models → Walk-forward OOS → Metrics/Calibration → Weakness Research → Candidate Validation → Adoption Gate → Registry → Production Prediction architecture. Current prediction is fail-closed when a valid adopted model/current snapshot is unavailable.

## Target isolation
Maintain independent semantics and experience for 1X2, Score Top1/Top3, O/U, BTTS and MOM Top1/Top4. Do not mix incompatible outcomes or retroactively change target definitions.

## Freshness/PIT
retrieved_at_utc is never historical availability evidence. Historical publication evidence is required before PIT verification. Current production requires a successful fresh matchday snapshot; stale prior artifacts are not current prediction evidence.

## Research
Features may use team strength/form, schedule/rest, venue, injuries/lineups, market context, event timing and source-quality metadata when PIT-safe. Rolling windows end at cutoff. OOS is chronological and target-specific calibration must be retained.

## Failure frontier
Prioritize stale-snapshot failures, late lineup information, high-confidence misses, unexplained ranking failures, source conflicts, OOD and competition distribution shift.

## Runtime safety
The research runner must fail closed when the preflight handoff environment is missing or invalid. TESTS_PASSED and AUDIT_PASSED are required explicit boolean inputs; absence or invalid values block research execution.