# Live Research Forecast Contract

## Purpose

This lane produces a current-state research forecast for explicitly targeted soccer matches. It is designed for situations where the adopted Production bundle is unavailable, blocked, or intentionally not being used, but a current match still needs a reproducible forecast artifact.

The lane does not grant Production status and must never bypass the Production adoption gate.

## Current architecture

Current UTC time
→ Sofascore live snapshot
+ Sofascore scheduled snapshot
+ current World Elo snapshot
→ source response hashes + retrieval timestamps
→ target-pair identity matching
→ current match state
→ Elo prior
→ Poisson score distribution
→ live remaining-goal adjustment when match is in progress
→ 1X2 + Top-3 scorelines
→ uncertainty + predictability + data completeness
→ PREDICT / DEFER / FAILED_CLOSED
→ GitHub Actions Step Summary + Artifact

## Target scope

The current emergency/live research target pair set is intentionally narrow:

- France vs Belgium
- Italy vs Turkey / Türkiye

The pair matcher normalizes Türkiye to Turkey for identity comparison. It does not merge unrelated competitions, teams, youth/women's sides, clubs or national teams.

## Time and PIT

Every forecast row records:

- kickoff_utc
- prediction_time_utc
- source_available_at_utc
- source_available_lower_bound_utc
- source_retrieved_at_utc
- source response hashes
- source event snapshot hash
- configuration hash
- Git SHA
- experiment fingerprint
- prediction revision ID

The key safety rule is:

retrieval time is not historical publication availability.

The live lane may use a current observation as a conservative current-state availability lower bound. It must not backdate that observation to claim historical publication availability.

For Production/OOS replay, unknown historical source availability remains unknown and therefore cannot silently become PIT-valid.

## Model

This is a transparent research heuristic, not a validated Production model.

Pre-match:

1. Read current World Elo ratings.
2. Add a configurable home-advantage Elo offset.
3. Convert relative strength into a home/away expected-goal split.
4. Use a Poisson joint score distribution.
5. Aggregate the joint distribution into Home/Draw/Away.
6. Apply conservative uniform probability shrinkage.

Live:

1. Start from the same pre-match expected-goal prior.
2. Observe the current score and elapsed time.
3. Estimate a bounded scoring-tempo ratio.
4. Scale the remaining expected goals within conservative limits.
5. Add the remaining score distribution to the observed score.
6. Aggregate final-score probabilities into 1X2.

No betting odds are used by this lane.

## Output contract

Principal artifact:

artifacts/live_research_predictions.csv

Status artifact:

artifacts/live_research_prediction_status.json

A row contains at minimum:

- match identity
- current status and current score
- elapsed time
- 1X2 probabilities
- three most likely final scorelines
- uncertainty
- predictability
- data completeness
- model status
- PIT status
- production status
- source snapshot hashes
- configuration hash
- Git SHA
- experiment fingerprint
- prediction revision ID

## Research safety

The lane is always marked:

production_status = NOT_PRODUCTION

The verification test rejects any row that attempts to mark itself as Production.

Failure behavior is fail-closed:

- missing configuration → FAILED_CLOSED
- unreachable or invalid source → FAILED_CLOSED
- missing target-team Elo → FAILED_CLOSED
- malformed target event → row-level failure; unaffected targets can still be evaluated
- no qualifying target event → NO_TARGET_EVENT_OR_BLOCKED
- empty output remains schema-bearing rather than becoming a misleading blank file

## Uncertainty

uncertainty is intentionally different from confidence.

The current research implementation includes:

1 - max(P(Home), P(Draw), P(Away))

plus a bounded data-completeness penalty.

predictability is entropy-derived and then discounted by data completeness.

These metrics are telemetry and research evidence. They are not yet calibrated claims.

## Reproducibility

A forecast can be grouped by:

config_sha256 + git_commit_sha + Elo snapshot hash + event snapshot hash + prediction state

to form its deterministic experiment_fingerprint.

prediction_revision_id is a compact identifier for a specific observed state and forecast revision.

## Automation

.github/workflows/soccer-live-research-forecast.yml

runs:

- on repository changes affecting the live lane
- every 5 minutes
- manually through workflow_dispatch

It has:

- bounded runtime
- set -euo pipefail
- dependency installation retry
- concurrency protection
- unit-contract verification
- forecast execution
- Step Summary publication
- short-lived artifact retention

## What this does not prove

A successful GitHub Actions run proves execution and output-contract compliance for this research lane.

It does not prove:

- chronological OOS superiority
- calibrated probabilities
- future generalization
- Production readiness
- robustness under distribution shift
- historical PIT replay validity
- adoption-gate satisfaction

Those claims remain governed by the main research/OOS/holdout/adoption system.

## Promotion path

No direct promotion exists.

Intended future path:

Prospective live snapshots
→ later result maturity
→ PIT/schema validation
→ match-level chronological WFO/OOS
→ prequential calibration
→ ablation
→ robustness
→ incumbent comparison
→ frozen holdout
→ adoption gate
→ shadow
→ promotion

Only evidence from that chain can justify Production use.
