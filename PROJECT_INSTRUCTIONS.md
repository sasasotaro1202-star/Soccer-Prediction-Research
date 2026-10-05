# PROJECT_INSTRUCTIONS — Soccer-Prediction-Research

## Mission
Target-specific soccer prediction with fresh matchday snapshots and separate contracts for 1X2, Score Top1/Top3, O/U, BTTS and MOM Top1/Top4.
Optimize Future Generalization, case-level correctness, probabilistic quality, calibration, uncertainty, predictability awareness, robustness, PIT integrity, information value, selective prediction and operational reliability. Historical fit alone is not success.

## Every run
Re-check the latest GitHub HEAD/default branch, code/config, tests, workflows, Actions, artifacts, registries, production/champion/challenger state, current research/OOS/holdout evidence, failures and backlog. Prefer REUSE → REPAIR → INTEGRATE → TEST → VERIFY. Never rewrite past evidence to make the current state look consistent.

## PIT / time
Separate event/market time, prediction cutoff, source availability, publication, retrieval, effective time and revision time. Only use information demonstrably available by cutoff. Unknown/unverifiable availability is fail-closed for production-quality OOS.

## Evaluation
Use chronological walk-forward OOS/WFO. Random splits are prohibited for temporal prediction. Separate candidate selection from final OOS. Frozen holdout is final evidence only and may not be tuned.

## Data integrity
Missing ≠ zero. Preserve explicit unavailable/unknown/delayed/not-yet-public/source-failed/malformed/not-applicable states. Preserve source lineage, snapshots, schema, revision behavior and identity history.

## Models / routing
Maintain simple baselines. Add challengers only for demonstrated incremental OOS value plus robustness. Specialist routing requires sufficient sample/folds/class coverage where relevant, calibration evidence and recent stability; otherwise fallback to broader validated scope.

## Calibration / uncertainty
Calibrate chronologically. Confidence ≠ predictability. Track disagreement, OOD, data/source uncertainty, regime ambiguity and event/volatility/liquidity uncertainty when relevant. Permit FALLBACK/ABSTAIN/DEFER/WAIT/ACQUIRE_MORE/RECOMPUTE instead of forced prediction.

## Research / adoption
External methods must pass discovery, source verification, PIT/cost checks, local reproduction, chronological OOS, robustness and frozen holdout before adoption. A candidate is never promoted from a single fold or external claim.

## Reliability / cost / security
Use checkpoint/resume, idempotency, bounded retry/backoff, watchdog/heartbeat, stale-run detection, deterministic writes, artifact preservation, concurrency control, recovery and rollback. Never hide failure. Prefer verified free/OSS/local/cache; unknown-cost or billing-risk services are not automatic dependencies. Never expose credentials.

## Completion
Green CI, generated artifacts or a completed workflow do not prove performance verification or production completion. Evidence must cover tests, PIT/leakage/meta-leakage, relevant universe/scope audits, chronological OOS/WFO, calibration, ablation, robustness, frozen holdout, artifact integrity, reproducibility, recovery, release gate, monitoring and rollback.

## Status
Use IMPLEMENTED / EXECUTED / VERIFIED / PERFORMANCE_VERIFIED / PROMOTION_CANDIDATE / ADOPTED / PRODUCTION / STABLE / HOLD / REJECTED / FAILED / BLOCKED / DEFERRED / ROLLED_BACK / UNKNOWN / UNVERIFIABLE / SUPERSEDED / RETIRED distinctly.

## Loop
MONITOR → DETECT → TRIAGE → RESEARCH → IMPLEMENT → TEST → PIT → OOS/WFO → CALIBRATION → ROBUSTNESS → HOLDOUT → ADOPT/HOLD/REJECT → RELEASE → PRODUCTION → RECONCILE → FAILURE ANALYSIS → MEMORY → NEXT RESEARCH.


## Research runner handoff
Research execution must receive explicit TESTS_PASSED and AUDIT_PASSED boolean states. Missing/invalid handoff is BLOCKED. Retry only bounded transient failures; deterministic implementation/schema failures fail fast and remain non-OOS.

## Feature ecology / feature-selection contract

Feature choice is a first-class research axis. Do not treat the full numeric feature matrix as the default production feature set merely because columns exist in the feature builder. The exact production-consumed `feature_cols` must come from the validated/locked model bundle and feature-selection record. If the current bundle cannot be verified, production feature usage is UNKNOWN/UNVERIFIABLE.

Feature states are:
- ACTIVE: explicitly included by a currently verified production bundle.
- CONDITIONAL: eligible only under an explicit PIT-safe context/cutoff contract.
- OBSERVATION_ONLY: collected/preserved for analysis or evidence and not a production model input.
- RESEARCH_CANDIDATE: allowed in research/OOS experiments but not production-authorized.

Feature selection is target-specific. 1X2 feature selection cannot be silently reused for Score, O/U, BTTS or MOM.

The candidate space must treat these as separate selection axes: feature family; individual feature/subset; historical window; representation (level/home-away/difference/other explicitly defined transform); aggregation (average/EWMA/volatility/trend); interactions; source/context channel; model; training window; calibration; routing policy.

Valid research patterns include baseline-only, additive family expansion, backward family ablation, individual/group deletion, window ablation, average-vs-EWMA comparison, home/away-vs-difference representation, interaction ablation, advanced-stat ablation, source removal, missingness-aware variants, model-specific selectors, competition specialists, and regime/data-quality conditional feature sets. Combinatorial search must be bounded by explicit candidate budgets and experiment fingerprints.

Feature selection is nested/prequential. For each outer chronological OOS fold, a data-driven selector may use only information available in the training side of that fold. When the selector itself learns from data, inner chronological WFO/validation inside the training side is required where necessary. The current outer OOS block is scored only after feature set, model, training window, calibration and routing are locked. Locked/frozen holdout data is never used to choose features, transformations, sources, thresholds, routing or model identity.

Feature-set identity is part of experiment identity. Record, at minimum, feature_set_id, ordered feature_cols, feature manifest/policy versions, source-set fingerprint, data snapshot, model id, training-window definition, calibration, router, OOS definition, seed, code SHA and experiment fingerprint. Equivalent candidates must be deduplicated rather than counted as independent experiments.

Model-window-feature selection must remain jointly prequential when these choices can influence one another. Sparse or unstable specialist evidence must fallback hierarchically to a broader validated feature/model configuration. A specialist is not production-eligible merely because it exists or wins one fold.

Feature value is measured by incremental information, not column count. Features from the same upstream/mirror/wrapper/republisher do not constitute independent information. Group/source ablations are preferred to uncontrolled feature accumulation. A feature family that improves mean OOS but harms calibration, newest blocks, high-upset cases, robustness or PIT integrity is not automatically adopted.

Case-level analysis is required for material feature changes. Inspect high-confidence misses, upsets, OOD cases, source conflicts, sparse-history cases, new/rare competitions and degraded-data cases. Outcome-driven analysis may diagnose weaknesses but must not use the final holdout to tune the feature set.

Lineup, injuries, suspensions, market, weather and other late matchday context remain CONDITIONAL/RESEARCH_CANDIDATE unless historical availability at the prediction cutoff is independently proven. Current-page or retrieval-time values are never retroactive historical PIT evidence.

Production bundle integrity must include the exact feature list/order, manifest/policy versions, schema hash, source snapshot/lineage, target version, calibration, router and fallback. Source-code presence of a feature is never evidence that Production consumes it.

Any change capable of changing the feature matrix or feature-selection result is evidence-affecting and requires TEST → PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout → adoption/release before production use.

## Cross-project mechanism transfer

The five-repository set remains Baseball-Prediction-System, BTC-Prediction-Research, 7-Sport-Prediction-Research, Soccer-Prediction-Research and Stock-Daily-Prediction-3000. Cross-project transfer is mechanism-level only.

From Baseball-Prediction-System, adopt the explicit Feature Manifest/Policy distinction between ACTIVE, CONDITIONAL, OBSERVATION_ONLY and RESEARCH_CANDIDATE, deterministic feature assembly, actual feature-count/schema-hash recording, and explicit gating for lineup/weather context.

From Stock-Daily-Prediction-3000, adopt nested/prequential selection, binding model and training-window identity during selection, contiguous prior-fold evidence requirements where applicable, and evidence-freshness rules that invalidate stale OOS after evidence-affecting changes.

From BTC-Prediction-Research, adopt feature/data/source lineage, experiment schema and fingerprints, source-independence grouping, hierarchical routing/fallback, uncertainty separation, selective actions, and retention of negative evidence.

From 7-Sport-Prediction-Research, adopt competition/phase-aware scope management, specialist routing with sample/fold sufficiency checks, scheduled experience/reconciliation discipline, and strict separation of active production lanes from research-only scope.

No domain-specific feature, model, metric result, threshold or production claim is transferred directly across projects. Every transferred mechanism follows DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → ADAPT → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → LOCAL_FROZEN_HOLDOUT → SHADOW → PROMOTE.


## GitHub-side autonomous operation contract
GitHub Actions should perform the routine control loop without requiring a new chat instruction each cycle:
MONITOR → RECONCILE → RETRY_TRANSIENT_FAILURE_ONCE → PIT_AUDIT → TEST → OOS/WFO → CALIBRATION/ROBUSTNESS → ADOPTION_GATE → SAFE_RELEASE.
Automatic merge is permitted only for an explicitly marked `automation_policy: hardening-safe-v1` PR whose merge state is CLEAN, every reported check is completed successfully, and the changed-file allowlist contains no production/model/data/artifact mutation paths. Automatic merge must use a final head-SHA recheck. Model, feature, calibration, target, production bundle, frozen holdout and performance-affecting research changes are never auto-merged by the controller unless a future policy explicitly authorizes them after separate evidence.
PIT replay remains scheduled and diagnostic; BLOCKED evidence is never promoted to PASS. Transient Action failures may receive one bounded failed-job rerun, but deterministic failures remain authoritative and blocking.


## GitHub-side autonomous operation contract

Routine repository operation is expected to run from GitHub Actions without requiring a new chat instruction each cycle:
MONITOR → RECONCILE → RETRY_TRANSIENT_FAILURE_ONCE → PIT_AUDIT → TEST → OOS/WFO → CALIBRATION/ROBUSTNESS → ADOPTION_GATE → SAFE_RELEASE.
Automatic merge is limited to explicitly marked `automation_policy: hardening-safe-v1` hardening PRs whose base is the current `main`, whose diff stays inside the hardening allowlist, whose checks are complete and successful, and whose final head-SHA race check remains clean. Model, feature, calibration, target, frozen-holdout, production-bundle and performance-affecting research changes remain outside the automatic-merge policy.


## GitHub-side state-based autonomy

Routine execution must not depend on an exact cron tick. The `Soccer Autonomous Orchestrator` runs every 30 minutes, reads the current `main` SHA, checks active/current-main and completed/current-main history, and dispatches only stale/missing lanes on `main`. It covers prediction refresh, matchday intelligence, experience settlement, research, discovery, PIT and coverage audits, scope, source probing, catalog, robustness and integrity lanes. Duplicate active runs are held. Every decision is persisted as an auditable Actions artifact.

This control plane cannot authorize production/model/feature/calibration/target changes, frozen-holdout access or performance claims. Automatic merging remains limited to hardening-safe control-plane paths with completed successful verification and final SHA race checks.

## GitHub-native continuous control-plane requirement

GitHub Actions is the persistent execution layer. Routine operation must remain autonomous after a chat session ends.

Required control-plane layers:
MONITOR → WATCHDOG → RECONCILE → BOUNDED RECOVERY → PIT/TEST → RESEARCH/OOS → CALIBRATION/ROBUSTNESS → ADOPTION GATE → SAFE RELEASE.

At minimum:
- Soccer Autonomous Orchestrator runs on a recurring schedule and performs state-based catch-up against the live main SHA.
- A separate control-plane watchdog independently detects missing/stale orchestrator execution and can restart it with bounded retries.
- Action Failure Recovery may retry transient workflow failures once, but never bypasses deterministic gates or changes evidence status.
- Legacy/old-SHA runs must not consume current-main dispatch capacity or suppress catch-up decisions for the current main branch.
- A single failed dispatch/state query must be recorded as HOLD and must not terminate reconciliation of later lanes.
- Every autonomous controller must use bounded API timeouts/retries, concurrency control, auditable decision artifacts, and fail-closed behavior.
- Model, feature, target, calibration, frozen-holdout and production changes remain outside autonomous hardening authority unless separately authorized by the adoption policy.

Continuous operation means the repository can resume after missed schedules, transient API failures, runner delays, or stale queued runs without requiring a new chat message.

## Long-term GitHub schedule keepalive

For public-repository long-term autonomy, maintain a low-frequency repository-activity heartbeat on GitHub Actions. The heartbeat may update only a dedicated automation-state file and must never touch model, feature, target, calibration, OOS, frozen-holdout or production evidence. Heartbeat commits should be excluded from heavy source/test workflows where safe, and heartbeat failures must use bounded recovery.

## Continuous prospective in-play research

For dynamic match-state research, when qualifying historical PIT snapshots do not exist, the project may build a prospective PIT dataset from a free/public current live source. Each snapshot must preserve observation/cutoff/availability/retrieval timestamps and an immutable source-response hash. Source availability may only be asserted at the conservative direct-observation boundary; it must never be backdated from a later page.

Prospective capture and label maturity are separate phases:
PROSPECTIVE OBSERVATION → LATER LABEL MATURITY → MATCH-LEVEL WFO/OOS → PREQUENTIAL CALIBRATION → ROBUSTNESS → INCUMBENT COMPARISON → FROZEN HOLDOUT → RELEASE GATE.

For the World Model track, market odds are not consumed. They must not become hidden features through source payloads, generic feature selection, or diagnostic fusion.

Prospective data acquisition is research evidence generation only. It cannot change target semantics, Champion, Production, frozen holdout or adoption authority.


## Current-match live research lane

A dedicated research-only current-match lane may serve explicitly requested live or imminent matches when the adopted Production bundle is unavailable or blocked. The lane must remain outside Production authority and must not relax PIT/OOS/holdout/adoption gates.

The live lane should:
- acquire a current public match snapshot;
- preserve response hashes and retrieval timestamps;
- keep retrieval time distinct from historical source publication/availability time;
- resolve target identity conservatively without silent merging;
- expose current score/status and elapsed time;
- produce 1X2 probabilities and a Top-3 score distribution when the current state is sufficiently observed;
- compute uncertainty and predictability separately;
- persist Git SHA, configuration hash, experiment fingerprint and prediction revision;
- publish to a dedicated research artifact and GitHub Actions summary;
- fail closed when critical sources or contracts are invalid;
- never label its own output PRODUCTION without passing the normal adoption contract.

Current live research implementation:
src/prediction/live_research_forecast.py

Current configuration:
config/live_research_forecast.json

Current workflow:
.github/workflows/soccer-live-research-forecast.yml

Current documentation:
docs/live-research-forecast.md

The intended maturation path is:
PROSPECTIVE SNAPSHOT → RESULT MATURITY → PIT/SCHEMA → CHRONOLOGICAL WFO/OOS → PREQUENTIAL CALIBRATION → ABLATION → ROBUSTNESS → INCUMBENT COMPARISON → FROZEN HOLDOUT → ADOPTION GATE → SHADOW → PROMOTION.

The emergency/live lane must never be interpreted as evidence that the underlying heuristic is performance-verified merely because a current forecast was generated successfully.
