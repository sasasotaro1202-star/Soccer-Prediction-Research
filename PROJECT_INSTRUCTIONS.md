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


## GitHub-side state-based orchestrator
The repository must not depend on exact cron timing for continuity. GitHub Actions runs a central `Soccer Autonomous Orchestrator` every 30 minutes and may catch up missing/stale routine lanes on `main`. It checks for active current-main runs before dispatching, respects per-lane minimum gaps, and records every DISPATCH/HOLD decision as an auditable artifact. It never grants performance claims, frozen-holdout access, or automatic production/model promotion.
