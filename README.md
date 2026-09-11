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
- `Soccer Research Cycle`: scheduled/manual acquisition, PIT gate, OOS research and artifact publication.

## Important current status

The initial Football-Data.co.uk adapter can acquire historical rows for seven mapped competitions, but it deliberately leaves `source_available_at_utc` unknown. The research engine therefore blocks PIT-verified OOS/model adoption until auditable source-availability timestamps are available. This is intentional and prevents retrospective leakage.

See `docs/MIGRATION_STATUS.md` for the legacy `soccer` repository migration classification.
