# Overnight production priority

## Objective
Maximize real-world robustness, not historical backtest fit. Never weaken a gate to make an Actions run green.

## Competition policy
Research coverage should be as broad as practical: major domestic leagues, major domestic cups, UEFA Champions League, Europa League, Conference League, and other meaningful competitions. A competition is production-eligible only after all required data, PIT, quality, chronological OOS, calibration, and adoption gates pass.

Competition catalog presence is not production approval. Unknown or unavailable PIT evidence must remain unknown; it must never be converted to zero, guessed, or inferred from retrieval time.

## Reliability policy
- CI and scheduled research use bounded timeouts and retries.
- Failed external acquisition must fail closed and preserve diagnostics.
- Partial artifacts are never promoted as complete results.
- Deterministic tests run before expensive research.
- Scheduled jobs must be safe to rerun and must not create overlapping audits.
- A green workflow means execution completed; production eligibility still depends on explicit gate artifacts.

## Prediction contract
- 1X2: calibrated Home/Draw/Away probabilities.
- Score: exactly 3 highest-probability scoreline candidates when the score model passes its eligibility gate.
- MOM: exactly 4 highest-probability Man-of-the-Match candidates when a valid pre-kickoff player universe and model pass their gates.
- If required information is unavailable or unverified, abstain rather than fabricate.

## Improvement priority
1. Fix correctness/test failures without weakening safety.
2. Improve PIT coverage and source-availability evidence.
3. Expand competition adapters only when provenance is auditable.
4. Integrate Score and MOM end-to-end with pre-kickoff-only features.
5. Improve calibration, drift robustness, and abstention using chronological OOS evidence.
6. Strengthen registry/model/schema integrity checks.
7. Optimize runtime only after correctness and production gates are preserved.
