# J1 Imminent Research Forecast Contract

## Purpose

This lane exists for imminent J1 matches when the adopted Production bundle is unavailable or blocked. It produces a reproducible, current-observation research forecast without weakening the Production adoption gate.

## Scope

The configuration targets J1 competition identifiers:

- jpn.1
- J1
- J1 League
- Meiji Yasuda J1 League

It discovers scheduled matches over the configured two-day horizon and can use ESPN as the keyless fallback.

## Model

The lane is explicitly research-only:

- current ESPN J1 standings-derived strength;
- configurable home-advantage offset on the internal strength scale;
- Poisson joint score distribution;
- uniform probability shrinkage.

Target outputs remain separated:

- 1X2;
- Score Top3;
- O/U 2.5;
- BTTS;
- MOM Top4 is explicitly abstained because no PIT-verified player model is active in this lane.

No betting-market odds are consumed by the forecast model.

## PIT

For current/imminent forecasting, the snapshot is observed before kickoff. source_retrieved_at_utc is recorded, while historical publication availability remains unknown. Retrieval time is never converted into historical PIT evidence.

Rows are always:

- prediction_state = PREMATCH_RESEARCH_FORECAST for scheduled future matches;
- production_status = NOT_PRODUCTION;
- pit_status = CURRENT_OBSERVED_PRE_KICKOFF.

This is sufficient for a current research observation but not for historical PIT replay/OOS validation.

## What it does not prove

A generated J1 forecast does not prove:

- future generalization;
- calibrated probabilities;
- chronological OOS superiority;
- robustness;
- frozen-holdout performance;
- production readiness;
- adoption.

The normal research chain remains mandatory before any production use:

prospective/current observation → result maturity → PIT/schema → chronological WFO/OOS → calibration → ablation → robustness → frozen holdout → adoption gate → shadow → promotion.

## Automation

Workflow: .github/workflows/soccer-j1-imminent-research-forecast.yml

It runs every 15 minutes and stores a short-lived research artifact. It is intentionally independent from the Production forecast workflow, so a missing Production model cannot be silently bypassed or relabeled as Production.