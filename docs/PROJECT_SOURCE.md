# Soccer-Prediction-Research — Project Source

## Current reconciliation — live repository state

- The authoritative source is the live GitHub `main` ref. The current HEAD SHA must be re-read from GitHub on every run and is intentionally not hardcoded in this section.
- Main currently includes the adoption-gate parity hardening that was accepted from PR #202. This is integrity/release-gate hardening only; it does not constitute a model, OOS, holdout, Champion, or Production performance result.
- PR #200 remains a research-only feature/model/configuration ecology lane. Its live head, base and Actions state must be re-read before accepting any result.
- Historical artifacts such as `run_status.json`, `test_status.json`, `audit_status.json`, `experience_*`, OOS reports, or model files are not current evidence merely because they are committed. Freshness and provenance must be checked against the live code/HEAD before use.
- No Champion/Production claim is authorized without current chronological OOS/WFO, PIT, calibration, robustness, frozen-holdout, provenance and release-gate evidence.

## Current-run rule

- Reconcile code, config, tests, workflows, Actions, artifacts, registries, production/champion/challenger, research/OOS/holdout, failures and backlog against live GitHub state at the start of every run.
- Do not rewrite historical evidence to make it consistent with current code.


## Current run reconciliation — 2026-10-04

- The previously recorded `28e74436f393f7df8691e9846e8017dcc076a938` commit is the reconciliation baseline only. The authoritative current `main` is the live GitHub ref and was re-checked at run start; its current HEAD must always be read from GitHub rather than copied into this historical note.
- The research-only feature/model/configuration matrix remains in PR #200 and is not Production, Champion, or performance-verified evidence.
- During reconciliation, stale duplicate PRs #192, #193, #197, #198 and #199 were closed as superseded because their substantive PIT/recovery/operational hardening is already represented on current main or by later reconstructed main updates. Historical evidence was not rewritten.
- PR #200 was strengthened with an additional function-boundary PIT check: feature research and WFO execution now require explicit prediction cutoff and feature-source availability provenance for verified rows and fail closed on missing/late timing evidence. This change remains research-branch-only until the PR is merged and verified.
- Long-running research workflows in PR #200 were changed to preserve in-flight runs instead of cancelling them when another refresh arrives; this is operational hardening only and does not constitute performance evidence.
- At reconciliation time, GitHub Actions for the latest research PR state were queued; no completed PASS/FAIL result or research artifact from that strengthened head was accepted as verified. This note is historical and must not override current Actions state. Local network execution is unavailable in this environment, so no local test result is substituted for GitHub Actions evidence.
- Production status remains unchanged: no verifiable committed `models/current` bundle, no new Champion adoption, and no new OOS/holdout performance claim.
- The current main HEAD now includes `63223a36a64f84c5ca859a3c8347dccd581a6f64`, a production feature-source provenance hardening commit. It binds bundle metadata to feature manifest/policy versions, ordered feature schema hash, source-registry lineage hash and target-definition version. This is integrity hardening only.
- PR #201 was closed as `SUPERSEDED` after current main acquired a stronger equivalent provenance implementation; its commits remain in GitHub history.
- PR #200 was retargeted to the current main HEAD `63223a36a64f84c5ca859a3c8347dccd581a6f64` so its research comparison is no longer based on the older reconciliation baseline. Its latest Ultimate Matrix run is still executing and cannot be treated as verified performance evidence yet.
- PR #202 is open for research-gate integrity hardening and currently has CI queued. It has no production authority.
- The currently committed `run_status.json` / `audit_summary.json` / `experience_*` artifacts inspected during this run predate the current HEAD and are therefore historical/stale relative to current code. They must not be treated as fresh verification for current main.

## Verified state 2026-10-04
Current GitHub main is authoritative and is re-checked each run; this section intentionally avoids embedding a self-invalidating commit SHA. Latest merged work hardens OOS PIT lineage boolean/cutoff validation, makes research preflight state strictly fail-closed, prevents no-op experience commits caused only by volatile timestamps, and hardens current matchday discovery with same-upstream SofaScore host fallback, bounded live HTTP retry behavior, and fail-closed source-outage handling. These changes are operational/data-integrity hardening only; no model, target, calibration, OOS, holdout, registry, or production prediction was changed. Matchday source state is now explicit: `COLLECTED` requires no discovery errors; `COLLECTED_WITH_ERRORS` is usable only when a successful discovery source is recorded and redundancy checks pass; empty/error states fail closed.
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

## PIT Replay Audit fail-closed boundary — 2026-10-04

The diagnostic PIT Replay Audit workflow now uses bounded dependency-install retries and distinguishes the canonical audit `BLOCKED` result (exit code 2) from unexpected implementation/runtime failures. Only exit code 2 is normalized to a successful diagnostic workflow step; other non-zero exits remain fatal. This prevents infrastructure/logic failures from being silently classified as evidence shortage. PIT remains a hard gate for OOS/model adoption, and this change does not alter production predictions, model selection, target semantics, calibration, OOS or frozen holdout evidence.

A dedicated regression module `tests/test_predictability_calibration_pit_provenance.py` covers post-cutoff source availability, post-cutoff retrieval, and impossible retrieval-before-source ordering. The research workflow now includes this regression module in its explicit test-file allowlist; current-main Actions must still pass before it is marked VERIFIED. Predictability research also treats a `prediction_pit_gate=PASS` row with invalid cutoff/maturity or optional provenance ordering as a hard PIT failure rather than silently dropping it.


## Feature ecology and selection source — verified design contract 2026-10-04

Feature selection is a first-class research dimension rather than an implicit consequence of the feature builder. The current source-level feature generator can expose a large candidate matrix, while different model consumers use different subsets. Current main does not contain a verifiable `models/current` production bundle, so the exact production `feature_cols` cannot be claimed from source code alone. This source therefore distinguishes the candidate feature pool from verified production usage.

### Canonical feature-state vocabulary

The Soccer feature contract uses four states:
- ACTIVE — explicitly consumed by a currently verified production bundle.
- CONDITIONAL — eligible only when an explicit PIT-safe context/cutoff contract passes.
- OBSERVATION_ONLY — preserved for diagnostics/evidence and not fed into production probabilities.
- RESEARCH_CANDIDATE — available to research/OOS experiments but not production-authorized.

The exact feature list and order consumed by Production must be recorded in the bundle and linked to a feature manifest/policy version. Feature names appearing in Python source are not evidence of production consumption.

### Current feature candidate pool

Under the full-field historical input schema, the current `src/features/soccer_features.py` design has a nominal model-input candidate pool of about 540 numeric/bool columns. The actual count is input-dependent because advanced-stat fields are optional and missing source channels remain unavailable rather than being fabricated.

The candidate pool consists of:
1. Match/strength state: neutral-venue state, home-advantage state, global Elo, competition Elo, expected Elo probability, dynamic Elo, shrunk competition Elo and associated gaps/interactions.
2. Schedule/rest: home/away rest hours, rest difference and strength-rest interaction.
3. Recent team form at windows 3/5/10/20. Each window contains 49 summary statistics per team, covering games, GF/GA, points, GD, win/draw/loss rates, average/EWMA form, volatility, venue mix, total-goal tendency, clean sheets, failed-to-score rate, and optional basic/advanced match statistics.
4. Home/away representation and explicit differences for each window. With the full field set this yields 123 columns per window and 492 columns across the four windows.
5. H2H last-five descriptors.
6. Momentum comparing 3-match versus 10-match states for points, GD, GF, GA and win rate, including home, away and differential forms.
7. Interaction features including attack-defense matchup, draw tension and strength/rest relationships.
8. History-support counts.
9. PIT metadata such as `pit_verified` and `feature_source_max_available_at_utc`, which are validation metadata and are excluded from model features.

The principal feature families are documented in `docs/FEATURE_MANIFEST.md`. Matchday lineup, injuries, suspensions, market and weather context are separate research/conditional channels and are not assumed to be part of the core production 1X2 feature vector without a PIT-safe adoption result.

### Feature-combination research matrix

The candidate space is explicitly multidimensional:
- feature family;
- individual feature or subset;
- 3/5/10/20 or future validated historical window;
- level versus home/away versus difference representation;
- average versus EWMA versus volatility/trend representation;
- interaction inclusion/exclusion;
- source/context channel;
- model family;
- training window;
- calibration;
- routing policy.

The intended research patterns include:
- baseline/Elo-only versus richer representations;
- additive family expansion;
- backward group ablation;
- individual/group deletion;
- window-only and window-combination ablation;
- average-vs-EWMA selection;
- home/away-vs-difference selection;
- basic-stat versus advanced-stat ablation;
- H2H add/remove;
- interaction add/remove;
- source removal;
- missingness-aware variants;
- model-specific feature selectors;
- competition specialists;
- regime/data-quality conditional feature sets.

This is a research search space, not a statement that every combination is currently implemented or production-authorized.

### Nested/prequential selection firewall

For an outer chronological OOS fold, feature selection may use only the training side of that fold. If the selector is data-driven, its own parameters/ranking/percentile must be learned from prior data, using an inner chronological validation/WFO process when selection requires tuning. The outer OOS block is scored once the feature-set, model, training window, calibration and routing identity are locked.

The locked/frozen holdout is never used for feature selection, feature transformation choice, source selection, threshold selection, model selection or routing. Selection evidence must be contiguous enough to support the claim being made; sparse prior-fold evidence cannot be silently treated as equivalent to a complete prequential sequence.

Feature-set identity is part of the experiment fingerprint. The minimum reproducibility record is:
`feature_set_id`, ordered `feature_cols`, manifest version, policy version, source-set fingerprint, data snapshot/hash, model id, training-window definition, calibration, router, OOS definition, seed, implementation Git SHA and experiment fingerprint.

When model, feature set and training window can influence one another, they must be selected as one jointly prequential candidate identity. This prevents the same chronological OOS information from being reused first to select a model/window and then to claim unbiased performance for the resulting feature configuration.

### Information value over feature count

A larger feature vector is not inherently better. Features derived from the same upstream, mirror, wrapper or republisher do not count as independent information. New feature families should therefore be evaluated by incremental OOS value and failure-slice behavior, with source/group ablations used to distinguish information gain from feature-volume effects.

Material feature additions must report at least OOS LogLoss, Brier, Accuracy, calibration/ECE, newest evaluation block, fold stability, robustness, PIT status and relevant case-level slices. High-confidence misses, upsets, OOD, source conflicts, sparse-history and degraded-data cases are explicit diagnostic slices. These slices are research diagnostics and cannot turn the frozen holdout into a tuning set.

### Feature-aware routing

A routed system may choose among validated feature/model configurations based on competition, phase, regime, data quality or other prediction-time states, but only when the routing evidence is chronologically validated. Specialist feature configurations require adequate sample size, fold/class coverage, recent stability and calibration evidence. Sparse or unstable specialists must fall back to a broader validated configuration.

### Production and research separation

Current research-only matchday context remains separate from the core production feature contract. In particular, current-page or retrieval-time lineup/injury/weather/market information is not historical PIT evidence. A conditional context feature becomes production-eligible only after explicit local PIT verification, chronological OOS/WFO, calibration, ablation, robustness, frozen holdout and adoption/release gates.

No feature-selection result, model-file existence or green Actions run is sufficient to change the Champion. The evidence state machine remains IMPLEMENTED → EXECUTED → VERIFIED → PERFORMANCE_VERIFIED → PROMOTION_CANDIDATE → ADOPTED → PRODUCTION, with HOLD/REJECTED/BLOCKED/UNVERIFIABLE preserved when evidence is insufficient.

## Cross-project mechanism review — 2026-10-04

The following mechanisms were rechecked in the other four repositories and transferred only at the mechanism level.

### Baseball-Prediction-System
Observed mechanism: a canonical Feature Manifest plus machine-readable feature policy distinguishes ACTIVE/CONDITIONAL/OBSERVATION_ONLY/RESEARCH_CANDIDATE; deterministic assembly and actual feature-count/schema-hash recording are explicit; lineup/weather are conditional context rather than automatic production inputs.

Soccer adaptation: use the same state vocabulary and manifest/policy pattern, while keeping Soccer's own target, feature schema and PIT rules.

### Stock-Daily-Prediction-3000
Observed mechanism: nested/prequential model-window ranking, contiguous prior-fold evidence requirements, explicit window-conditioned identity, and freshness rules that invalidate stale OOS evidence after evidence-affecting changes.

Soccer adaptation: feature-set, model and training-window selection are jointly prequential when coupled; a changed feature-selection implementation retriggers the relevant OOS lane.

### BTC-Prediction-Research
Observed mechanism: feature-level lineage, source graph/independence grouping, experiment fingerprints, hierarchical routing/fallback, selective actions, uncertainty separation and retention of negative research evidence.

Soccer adaptation: track feature/source lineage and experiment identity, avoid counting correlated mirrors as independent sources, and permit fallback/abstain/defer rather than forcing degraded feature configurations.

### 7-Sport-Prediction-Research
Observed mechanism: broad competition/phase scope management, specialist routing only with sufficient evidence, scheduled experience/reconciliation controls, and explicit production-versus-research lane separation.

Soccer adaptation: competition and phase remain explicit routing/evaluation dimensions, while all newly scoped competitions remain research-only until local evidence passes.

### Transfer firewall

No domain-specific feature, model, benchmark result or production claim is copied from another repository. Every imported mechanism is subject to:
DISCOVER → ABSTRACT_MECHANISM → COMPATIBILITY → ADAPT → LOCAL_PIT → LOCAL_OOS/WFO → ROBUSTNESS → LOCAL_FROZEN_HOLDOUT → SHADOW → PROMOTE.

## Current status after this governance update — 2026-10-04

This update is governance/documentation and feature-contract work only. It does not alter prediction probabilities, target semantics, calibration outputs, OOS results, frozen holdout, Champion, Production registry or historical experience.

Current main remains BLOCKED by recorded preflight test/data-audit failures; the repository still records no OOS claim from that blocked run. Current main also has no committed `models/current` production bundle or durable experience ledger that would allow an exact production feature list to be independently verified. Therefore the feature contract is updated and the candidate space is defined, but no new feature set is promoted or claimed as Production.




## GitHub-side autonomous operation — active on main after merge

The repository control plane is designed to operate from GitHub Actions without a fresh chat instruction each cycle. The autonomous controller only merges explicitly marked hardening-safe PRs after current-main/base equality, same-repository checks, allowlisted changed paths, completed successful checks, PIT evidence when relevant, and a final head-SHA race check. Production/model/feature/calibration/target/frozen-holdout/performance changes remain outside this automatic merge policy. The controller emits an auditable artifact and never converts missing or ambiguous evidence into PASS.


## GitHub-native autonomous orchestration — 2026-10-05

The live repository contains a state-based `.github/workflows/soccer-autonomous-orchestrator.yml` that reconciles routine soccer research and operations on a 30-minute cadence. It dispatches catch-up executions only on the current `main` SHA, refuses to suppress cadence based on old/PR runs, avoids duplicate active current-main runs, and records DISPATCH/HOLD evidence. Covered lanes include matchday refresh, daily forecast, 9H research, robust/predictability/historical research, experience settlement, adaptive discovery, scope frontier, PIT replay/coverage/versioned audits, global data lake, MOM, catalog/source audits, research cycle, overnight integrity and Opta-like monitoring. The orchestrator has no authority to change production models, frozen holdout evidence, targets, calibration, or performance claims.

The autonomous merge controller separately allowlists the orchestrator workflow and the narrowly scoped experience commit guard as hardening-safe-v1. This is a control-plane bootstrap only; research/performance-affecting changes remain outside automatic merge. All PIT/OOS/WFO/calibration/robustness/holdout/adoption gates remain authoritative.


## Current live-state reconciliation — 2026-10-05

- Live GitHub `main` was re-read on 2026-10-05 and remains the authoritative branch. The current HEAD is intentionally not hardcoded in this historical reconciliation; every new run must read the live `main` ref again.
- Current main has no verifiable `models/current/production_model.json` or `models/current/production_model.pkl`, and no committed current `model_registry.json`, `adoption_decision.json`, locked OOS metrics, calibration gate or production provenance artifact. Therefore Champion/Production performance remains UNKNOWN/UNVERIFIABLE, not zero and not a claimed regression.
- The latest committed research-cycle artifact state remains historical `BLOCKED` evidence with `oos_claimed=false`; it is preserved and is not rewritten. No newer accepted OOS/holdout performance evidence is currently committed on main.
- PR #215 (`fix: restore research engine file hashing helper`) has been merged to main after its exact-head `Soccer CI` completed successfully. The change restores the deterministic `_sha256` helper and adds regression coverage; it changes no prediction/model/OOS/holdout policy.
- PR #218 (`hardening: bound autonomous soccer dispatch pressure`) is now the clean current-main replacement for superseded #214/#217. It is based directly on the current main and contains only the reviewed dispatch backpressure workflow/test change; its exact-head `Soccer CI` is queued/in progress and it has not been merged.
- PR #212 adds dynamic match-state hazard research and remains research-only; its successful research workflow does not authorize performance adoption because qualifying historical in-play PIT evidence is absent.
- Scope Frontier currently reports `PARTIAL`: the 2026-10-05 through 2026-10-18 discovery window contains zero newly discovered candidates and repeated HTTP 403 failures from the current SofaScore scheduled-events endpoint. This is acquisition/coverage evidence, not model-performance evidence.
- No new OOS/WFO, calibration, robustness, frozen-holdout or adoption result was accepted during this reconciliation. No historical performance evidence was modified.

## Current live-state reconciliation — 2026-10-05 (post control-plane hardening)

- Live main was re-read after the latest control-plane changes. This section is a new reconciliation record; prior historical status statements are preserved unchanged.
- The autonomous orchestrator on main now includes bounded dispatch retries, per-lane failure isolation, current-main-only active-run accounting, and a dispatch budget. A failed lane is recorded as HOLD and does not terminate the remaining reconciliation lanes.
- Action Failure Recovery now includes the Soccer Autonomous Orchestrator workflow as a bounded one-time recovery target. Quality/PIT gates remain authoritative.
- A separate .github/workflows/soccer-control-plane-watchdog.yml is active on a 10-minute schedule. It checks the latest main SHA, detects a missing/stale orchestrator run, restarts a stuck current-main orchestrator, and retries control-plane API dispatches with bounded backoff. It cannot modify models, targets, calibration or holdout evidence.
- PR #219 was closed without merge because its hardening changes were already promoted directly to the current main through controlled hardening commits. No research-performance claim was introduced by that promotion.
- Current production/champion performance remains UNKNOWN/UNVERIFIABLE. The repository still has no newly verified OOS/WFO/calibration/robustness/frozen-holdout adoption evidence from this control-plane work.
- The historical BLOCKED run artifacts remain preserved. They are not rewritten into PASS and do not count as new OOS evidence.

## Current live-state reconciliation — 2026-10-05 (event-driven watchdog)

- The control plane now has both scheduled and event-driven liveness checks. The 10-minute watchdog schedule remains active and the same watchdog also runs after completion of key orchestrator/research/PIT workflows.
- Watchdog decisions are persisted as a 14-day Actions artifact. Current-main SHA is resolved live on every run; missing/stale orchestrator execution is recovered with bounded API retries.
- Action Failure Recovery now includes timed-out control-plane runs in its one-time bounded recovery path. Deterministic failures and research quality gates remain authoritative.
- No model, target, calibration, frozen-holdout or production performance evidence was changed by this hardening.

## Current live-state reconciliation — 2026-10-05 (controller queue hardening)

- Autonomous GitHub Controller now uses cancel-in-progress concurrency so bursts of workflow_run events do not accumulate stale controller executions. The newest reconciliation request supersedes older queued controller runs.
- The hardening allowlist includes the independent control-plane watchdog, so future explicitly marked hardening-safe watchdog changes remain eligible for the same gated auto-merge path.
- Current main remains authoritative and all hardening changes above are control-plane only; no prediction probabilities, feature selection, model identity, calibration, target definition or frozen-holdout evidence has been altered.

## Current live-state reconciliation — 2026-10-05 (long-term schedule keepalive)

- A dedicated weekly `Soccer Automation Heartbeat` workflow now creates deterministic activity under `.github/automation/heartbeat.json` using bounded GitHub API retries. This is a control-plane keepalive only and cannot change prediction/model/feature/target/calibration/frozen-holdout evidence.
- Heartbeat-only commits are excluded from the heavy Soccer CI and Legacy V9/V12 Bridge push triggers, preventing the keepalive itself from creating unnecessary research execution load.
- Heartbeat failures are included in bounded Action Failure Recovery.
- GitHub documents that public-repository scheduled workflows can be automatically disabled after 60 days without repository activity; the weekly heartbeat is intended to maintain repository activity while the scheduled control plane remains enabled. GitHub also documents that scheduled events can be delayed under high load, so the independent watchdog remains the primary short-latency recovery layer.

## Current live-state reconciliation — 2026-10-05 (failure backoff)

- The autonomous orchestrator now applies bounded exponential backoff to consecutive exact-main failures/timed-out/cancelled executions of a lane. The cooldown grows up to 24 hours, while successful current-main history restores normal cadence evaluation.
- This reduces deterministic-failure thrashing and protects GitHub runner/free-usage efficiency without converting failure evidence to PASS.
- No model, feature, target, calibration, OOS, frozen-holdout or production evidence is modified by this control-plane hardening.

## Hierarchical Dynamic Simulator frontier — 2026-10-05

The supplied hierarchical Bayesian dynamic football simulator blueprint is registered as a research frontier at `docs/HIERARCHICAL_DYNAMIC_SIMULATOR.md` with machine-readable policy at `config/dynamic_simulator_policy.json`.

The intended mathematical ladder is:

- verified Score distribution foundation;
- hierarchical latent team strength;
- dynamic match-state hazard;
- probability-tree/scenario propagation;
- lineup distribution;
- player/role interaction;
- full posterior-predictive event simulation.

The frontier runner is `src/research/dynamic_simulator_frontier.py`. It is research-only and fail-closed. It may report WARMUP/HOLD/READY states but cannot authorize Production, Champion, frozen-holdout access, or performance claims.

The GitHub lane `.github/workflows/soccer-dynamic-simulator-research.yml` runs on a recurring six-hour schedule and is also caught up by the 30-minute autonomous orchestrator. It records research evidence as workflow artifacts and keeps Production immutable.

The lane may bridge the existing open dynamic match-state research candidate (PR #212) by dispatching its own research workflow when that branch contains the required workflow file. This is research execution only; it does not merge, promote, alter the Champion, or touch frozen holdout.

The mathematical simulator remains staged. Complexity is added only when identical chronological OOS evidence demonstrates incremental value relative to the incumbent, with PIT, calibration, robustness, event dependence, reproducibility and holdout controls preserved.
