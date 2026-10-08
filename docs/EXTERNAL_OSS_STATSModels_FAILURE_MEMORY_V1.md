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
