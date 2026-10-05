# Soccer-Prediction-Research — Project Instructions

## Authority and every-run reconciliation
- `docs/PROJECT_SOURCE.md` and live GitHub `main` evidence are authoritative. Read the current `main` SHA on every run; never substitute an old conversation snapshot.
- Before any work, reconcile: Project Instructions, Project Source, HEAD/branch, code/config/tests/workflows/Actions/artifacts, Champion/Challenger/Production, research/OOS/holdout, failures and backlog.
- Preserve historical evidence. Do not rewrite old results, holdouts, failures or stale artifacts merely to match current code.

## Objective
- Optimize for future generalization, case-level correctness, calibration, uncertainty/predictability awareness, robustness, PIT integrity, information efficiency, recovery reliability, reproducibility and operational safety — not historical accuracy alone.

## Target isolation
- Keep 1X2, Score Top1/Top3, O/U, BTTS and MOM targets independent.
- Target semantics and versions are immutable for historical evidence; a target change creates a new target version.

## PIT / temporal integrity
- Track at minimum: `match_id`, `kickoff_time`, `prediction_time`, `prediction_cutoff`, `source_available_at`, `published_at`, `retrieved_at`, `revision_time`.
- Require `source_available_at <= prediction_cutoff` for historical evidence. `retrieved_at` is not historical availability proof.
- Unknown, missing or ambiguous PIT is FAIL-CLOSED/HOLD; never turn unknown into valid.
- Same-match information must be processed in kickoff order when result-derived evidence could leak across matches.

## Data and lineage
- Maintain source registry, snapshot hashes, feature lineage, available-at/revision lineage, schema/parser versions, implementation Git SHA and experiment fingerprint.
- Mirrors/wrappers/republishers of the same upstream are not independent sources.
- Stable IDs are required for competition, season, phase, team, player and match. Fuzzy matching is review-only; silent merges are forbidden.
- Missing values remain missing/unknown unless a PIT-safe, validated imputation policy explicitly exists.

## Research and evaluation
- Architecture: Data Acquisition → Coverage → QC/PIT → Features → Candidate Models → chronological WFO/OOS → Metrics/Calibration → Weakness Research → Candidate Validation → Adoption Gate → Registry → Production Prediction.
- Random train/test splitting is prohibited for performance claims; chronological WFO/OOS is the default.
- Candidate selection, final OOS and frozen holdout are strictly separated.
- Primary 1X2 metric = LogLoss; secondary = Accuracy, Brier, class-wise metrics and calibration/ECE. Score uses distribution-aware metrics including MAE/top-k evaluation.
- Calibration is evaluated independently from accuracy and cannot use the frozen holdout for tuning.

## Matchday intelligence and uncertainty
- Use lineup, injuries, suspensions, formation, schedule/rest, travel, venue, weather, market context and official late updates only when PIT-safe at the prediction cutoff.
- Distinguish confidence from predictability.
- Where justified, choose PREDICT, ACQUIRE_MORE, WAIT, RECOMPUTE, FALLBACK or ABSTAIN.
- Inspect model disagreement, OOD, regime ambiguity, data completeness, source disagreement, volatility, lineup uncertainty and calibration instability.

## Robustness and failure memory
- Test PIT attacks, leakage/meta-leakage, time shift, feature deletion, source removal, missingness, outliers, distribution shift, regime changes, unseen teams/players/competitions, upset periods and stale sources.
- Record failures as data/PIT/identity/feature/model/calibration/timing/regime/OOD/uncertainty/scope/automation classes and close the loop: failure → hypothesis → experiment → validation → adopt/hold/reject.

## Production and adoption gate
- Code existence, green CI, artifact existence or a passing single run never equals performance verification.
- Champion/Production authority requires fresh chronological OOS/WFO + PIT + calibration + robustness + frozen-holdout + provenance + release-gate evidence.
- Reference adoption thresholds may include relative OOS LogLoss ≥3%, auxiliary ≥1%, no deterioration in ≥70% of evaluation blocks, no newest-holdout deterioration, materially stable calibration and zero PIT violations; use variance, confidence intervals, dependence, rare events and operational cost before adoption.
- Frozen holdout is final-evaluation-only and must never be used for tuning or scope selection.

## GitHub-native continuous control plane
Operate the repository continuously through:
MONITOR → WATCHDOG → RECONCILE → BOUNDED RECOVERY → PIT/TEST → RESEARCH/OOS → CALIBRATION/ROBUSTNESS → ADOPTION GATE → SAFE RELEASE.

Current control-plane responsibilities:
- `Soccer Autonomous Orchestrator`: state-based catch-up on a 30-minute cadence against the live `main` SHA.
- `Soccer Control Plane Watchdog`: independent 10-minute liveness/restart protection plus event-driven checks.
- `Soccer Automation Heartbeat`: weekly repository-activity keepalive using bounded retries.
- `Soccer Automation Integrity Audit`: recurring verification that the control-plane files, schedules, safety flags and liveness assumptions remain intact.
- Action Failure Recovery may perform only bounded retries; a failure is never rewritten into success.
- Old-SHA runs must not block current-main catch-up. One failed lane must not terminate later lanes.
- Automation hardening must not mutate models, targets, calibration, frozen holdout, historical evidence or performance claims.

## Autonomous change policy
- Safe/free code, tests, documentation, workflow hardening and recovery improvements may be implemented directly on GitHub without repeated micro-confirmation.
- Performance-affecting model/feature/target/calibration/production changes remain behind the normal PIT, chronological OOS/WFO, robustness, frozen-holdout and adoption gates.
- Paid-only, billing-risk or unknown-cost services remain HOLD/UNCONFIRMED.
- Secrets, tokens, credentials and personal data must never be written to source or artifacts.

## Automation reliability
- Prefer checkpoint/resume, idempotency, bounded timeout/retry/backoff, watchdogs, heartbeats, stale-run detection, deterministic writes, concurrency control, artifact preservation and rollback/recovery.
- Never use `|| true` or equivalent to hide required failures.
- Status must distinguish IMPLEMENTED, EXECUTED, VERIFIED, PERFORMANCE_VERIFIED, PROMOTION_CANDIDATE, ADOPTED, PRODUCTION, STABLE, HOLD, REJECTED, FAILED, BLOCKED, DEFERRED, ROLLED_BACK, UNKNOWN, UNVERIFIABLE, SUPERSEDED and RETIRED.
