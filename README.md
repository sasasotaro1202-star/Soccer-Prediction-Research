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


## Total research forecast coverage
Daily and manual matchday research forecasts use total-coverage routing: when an adopted production model is unavailable or produces no eligible prediction rows, the workflow generates clearly labeled research-only fallback probabilities. Production remains fail-closed, and fallback outputs never alter the production registry.
