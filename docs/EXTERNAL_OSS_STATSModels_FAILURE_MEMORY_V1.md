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
