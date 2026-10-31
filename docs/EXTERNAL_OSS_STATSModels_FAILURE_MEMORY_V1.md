# Statsmodels External OSS Failure Memory v1

## Scope
Research-only external OSS challenger: `statsmodels/statsmodels`.

## Failure F1
- failure_type: `OOS_PIT_BOUNDARY`
- observed_at: `2026-10-08`
- workflow: `External OSS Statsmodels Soccer OOS`
- input: `artifacts/pit_replay_features.csv`
- PIT input audit: `42,547` total rows, `17,557` PIT-verified rows
- symptom: first chronological OOS fold had `999` PIT-safe training rows while `min_train=1000`
- root_cause: initial row-count boundary was fixed before applying the predictor-side availability cutoff
- unsafe repair rejected: lowering `min_train`
- repair: advance the chronological OOS boundary until the PIT-safe training prefix reaches the locked minimum; never use future OOS outcomes
- additional repair: advance boundaries past identical kickoff timestamps so one event-time cohort cannot be split across train/OOS
- verification required: Research tests and a fresh real-data OOS run on the repaired HEAD
- current state: `REPAIR_APPLIED_PENDING_REVERIFICATION`

## Failure F2
- failure_type: `RESEARCH_TEST_CONTRACT`
- symptom: two tests expected weaker filtering behavior after fail-closed PIT hardening
- root_cause: tests were not updated to the stricter contract
- repair: tests now require explicit failure for missing PIT training timestamps and align error matching with the availability contract
- current state: `REVERIFICATION_PENDING`

## Policy
Failures remain immutable research knowledge. A failed experiment is not converted to PASS by changing the evidence semantics. Any new result must use a new run identity and the exact code/data configuration that produced it.


## Failure F3
- failure_type: `MODEL_SUPPORT_OOD`
- observed_at: `2026-10-08`
- workflow: `External OSS Statsmodels Soccer OOS`
- symptom: incumbent Score model raised `Score model has no PIT-trained rate for one or both fixture teams` during chronological OOS
- root_cause: the OOS block contained a team/competition outside the support of the training prefix
- unsafe repair rejected: imputing an arbitrary score rate or silently counting an unavailable prediction as a correct/incorrect outcome
- repair: construct an explicit common evaluability set across incumbent and statsmodels challenger; exclude unsupported cases from performance metrics while retaining their count as `unsupported_rows` and reporting `common_coverage`
- interpretation: unsupported/OOD coverage is separate from model accuracy and is itself monitored evidence
- verification required: fresh Research tests and real-data OOS on the repaired HEAD
- current state: `REPAIR_APPLIED_PENDING_REVERIFICATION`



## Failure F4
- failure_type: `RESEARCH_TEST_THRESHOLD_DRIFT`
- observed_at: `2026-10-08`
- workflow: `External OSS Statsmodels Research`
- symptom: latest strengthened OOS contract caused 7 synthetic tests to fail before exercising the intended assertions
- root_cause: test fixtures still used the earlier `min_train=24` / small calibration assumptions while the locked research runner requires `min_train>=200` and `calibration_min_rows>=50`
- unsafe repair rejected: weakening the research runner thresholds
- repair: enlarge synthetic fixtures and explicitly pass the test-only calibration window while preserving the real-data thresholds
- additional repair: robustness synthetic coverage now requires distinct competition identities, matching the multi-competition robustness contract
- current state: `REPAIR_APPLIED_PENDING_REVERIFICATION`

## Failure F5
- failure_type: `ACTIONS_CANCELLATION_DUE_TO_PR_CHURN`
- observed_at: `2026-10-08`
- workflow: `Soccer CI / External OSS Statsmodels Research / External OSS Statsmodels Soccer OOS / Soccer Prospective In-Play Maturity`
- symptom: multiple runs for the same PR head were cancelled as the branch was updated repeatedly
- root_cause: PR-scoped concurrency and rapid sequential commits created overlapping verification runs
- interpretation: cancellation is an operational verification failure, not prediction-performance evidence
- repair: stop code churn before final verification; directly re-run the latest cancelled research jobs rather than treating cancellation as success
- current state: `REVERIFICATION_IN_PROGRESS`


## Failure F6
- failure_type: `FEATURE_PIT_AUTHORITY_TEST_MISMATCH`
- observed_at: `2026-10-08`
- workflow: `External OSS Statsmodels Research`
- symptom: latest verification failed 3 tests after tightening predictor-side PIT authority
- root_causes:
  - adapter rejected invalid row-level feature availability only by filtering, while the regression test expected fail-closed rejection
  - OOS runner retained a residual `source_available_at_utc` filter after `feature_source_max_available_at_utc` became the sole predictor-PIT authority
  - case-level PIT regression test still mutated the legacy source timestamp instead of the authoritative feature-source timestamp
- repair:
  - reject any PIT-verified row whose feature-source availability is after its own historical prediction cutoff
  - remove all legacy `source_available_at_utc` predictor-PIT references from the OOS runner
  - mutate the authoritative feature-source field in the case-level regression test
  - preserve `source_available_at_utc` as non-authoritative metadata only
- current state: `REPAIR_APPLIED_PENDING_REVERIFICATION`
- performance_claim: `false`
- production_adoption: `false`
- frozen_holdout_access: `false`
