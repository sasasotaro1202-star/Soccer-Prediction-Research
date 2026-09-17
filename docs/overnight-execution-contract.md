# Overnight execution contract

Routine overnight work may continue without interactive approval when it is limited to safe, deterministic research/CI improvements.

## Non-negotiable safety
- Never weaken PIT, leakage, chronological OOS, calibration, stability, provenance, or adoption gates merely to make a workflow green.
- Never directly replace or mutate the incumbent production model because a candidate looks better on a convenient split.
- External/X-data remains acquisition/adapter/PIT/offline-evaluation only until it passes the same-snapshot, same-PIT-cutoff, chronological OOS adoption gate.
- Any unverifiable or PIT-unsafe evidence is excluded from production eligibility.
- Preserve fail-closed behavior: infrastructure/network failure is a degraded/blocked state, not a successful research result.
- No Work mode.

## Reliability
- Use bounded retries and deterministic caching/deduplication where they preserve semantics.
- Cancel stale long-running research runs when a newer commit supersedes them; do not let stale jobs block newer validation.
- Keep archive verification conservative: exact fixture identity, explicit publication lower bounds, earliest qualifying capture.
- Optimize repeated archive work by grouping identical source URLs/captures, but never reduce evidence requirements.

## Evaluation
- Candidate changes must be compared with the incumbent on the same data snapshot, PIT cutoff, and chronological OOS folds.
- Track log loss, Brier score, calibration/ECE, fold stability, drift, and stress behavior rather than accuracy alone.
- Keep a locked final OOS evaluation separate from model-selection data.
- Adopt only when all existing production gates pass; otherwise retain the incumbent and record the blocker.

## Autonomous progress
- Inspect current Actions and artifacts before making changes.
- Prefer small, testable commits.
- Run deterministic tests and relevant audits after changes.
- Record useful provenance and failure causes so the next run can resume without guessing.
- Continue improving efficiency and production robustness while preserving the above contracts.
