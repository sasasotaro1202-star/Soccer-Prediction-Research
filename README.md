# Soccer-Prediction-Research

Chronological OOS soccer prediction, backtesting, calibration and automated research system.

## Current architecture

`Data Acquisition → Coverage → QC/PIT → Features → Candidate Models → Walk-forward OOS → Metrics/Calibration → Weakness Research → Candidate Validation → Adoption Gate → Registry → Production Prediction`

## Safety rules

- Point-in-time data only for production research.
- `retrieved_at_utc` is never treated as `source_available_at_utc`.
- Random train/test splits are prohibited.
- Candidate selection and final OOS are separated.
- OpenAI is advisory only; it cannot promote a model.
- Production prediction fails closed without an adopted registry model.
- Missing data is not converted to zero.

## GitHub Actions

- `OpenAI API Check`: validates the configured GitHub Actions secret and API connectivity.
- `Soccer Research Robust`: scheduled/manual acquisition, deterministic tests, structural audits, strict PIT/completion gate, bounded research retries, production-contract evaluation, health artifacts, and cached research evidence.

## Important current status

The current system uses a fail-closed PIT replay layer with archived evidence recovery, bounded retries, exact completed-result identity matching, and persisted PIT evidence caching between robust runs. Valid publication evidence is required before a row can be PIT-verified; unavailable evidence remains unknown rather than being imputed. Production adoption remains blocked unless the chronological OOS, stability, calibration, provenance, and PIT gates all pass.

## Latest prediction contract

On-demand/production predictions must use the freshest available matchday snapshot. The production runner requires a successful current matchday refresh and rejects snapshots older than 15 minutes, so previously generated prediction artifacts are not treated as current predictions.

Performance monitoring is target-specific. Durable experience metrics are stored separately for 1X2, Score Top1/Top3, O/U, BTTS, and MOM Top1/Top4, with probability-quality metrics retained where the required probabilities are available.

See `docs/MIGRATION_STATUS.md` for the legacy `soccer` repository migration classification.

## Continuous World Model research

The repository now contains a research-only continuous World Model path:

Prospective In-Play Capture → Later Maturity → PIT/Schema → Match-Level WFO/OOS → Prequential Calibration → Robustness/Ablation → Incumbent Comparison → Frozen Holdout → Release Gate.

The prospective collector uses only free/public current soccer state and records point-in-time provenance conservatively. It does not use betting odds. Missing or ambiguous observations are not converted to zero.

GitHub Actions provides the persistent execution layer through the prospective capture, maturity, dynamic simulator, World Model supervisor, autonomous orchestrator, watchdog and bounded failure-recovery lanes. These lanes can catch up after missed schedules or transient failures, but research infrastructure cannot self-promote into Production.

The dynamic simulator remains a research candidate until future chronological OOS, calibration, robustness, incumbent comparison and holdout evidence demonstrate incremental generalization.


## Current-match live research forecast

The repository also provides a separate current-match research lane:

.github/workflows/soccer-live-research-forecast.yml

It runs every five minutes and can be run manually. It targets the explicitly configured France–Belgium and Italy–Turkey/Türkiye pair identities, reads current public match state plus a current World Elo snapshot, and publishes a provenance-rich live research forecast.

The lane writes:

- artifacts/live_research_predictions.csv
- artifacts/live_research_prediction_status.json

The output includes 1X2 probabilities, Top-3 scorelines, current score/status, uncertainty, predictability, data completeness, PIT metadata, source hashes, configuration hash, Git SHA and experiment fingerprint.

This lane is always NOT_PRODUCTION. It does not bypass the adopted-model registry, OOS, calibration, robustness, frozen-holdout or adoption gates.

See docs/live-research-forecast.md for the complete contract.
