# Soccer-Prediction-Research — Project Instructions

## Priority
`docs/PROJECT_SOURCE.md` and current GitHub evidence are authoritative. Preserve historical results, source audits, OOS and holdout records.

## Target isolation
Keep 1X2, Score Top1/Top3, O/U, BTTS and MOM targets separate. Target semantics, experience ledgers and calibration must not be mixed or retroactively changed.

## PIT and production
Historical publication/availability must be evidenced; retrieval time alone is not proof. Rolling windows end at cutoff. Fresh matchday snapshots are required for current production; stale prior artifacts are not current prediction evidence. Identity, scope and PIT failures fail closed.

## Mandatory loop
MONITOR → DETECT → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS.

## Evaluation
Prioritize future generalization, probabilistic quality, calibration, case-level diagnostics, predictability and uncertainty. Evaluate competition/phase, regime, source outage/removal, feature deletion, OOD and recent periods.

## Cross-project transfer
Mechanisms from Baseball, BTC, 7-Sport and Stock are candidates only. Require local PIT, chronological OOS/WFO, robustness and frozen holdout before promotion.

## Failure and cost
No silent exception, fabricated metric, missing→zero, unknown PIT→valid or failed recovery→success. Unknown/billing-risk/paid paths remain HOLD/UNCONFIRMED.