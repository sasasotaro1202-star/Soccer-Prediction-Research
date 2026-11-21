import pandas as pd
import pytest

from src.research.external_oss.score_distribution_calibration import (
    fit_temperature,
    temperature_transform,
)


def _distribution():
    return [(0, 0, 0.8), (1, 0, 0.1), (0, 1, 0.1)]


def test_temperature_transform_normalizes():
    out = temperature_transform(_distribution(), 2.0)
    assert sum(p for _, _, p in out) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        temperature_transform(_distribution(), 0)


def test_temperature_fit_improves_or_preserves_training_calibration_objective():
    class FixedModel:
        pass

    def predict(model, home_team, away_team, competition, *, max_goals):
        return _distribution()

    frame = pd.DataFrame([
        {"home_team":"A","away_team":"B","competition":"T","home_goals":0,"away_goals":0},
        {"home_team":"A","away_team":"B","competition":"T","home_goals":0,"away_goals":0},
        {"home_team":"A","away_team":"B","competition":"T","home_goals":1,"away_goals":0},
    ])
    result = fit_temperature(frame, FixedModel(), predict)
    assert result["optimization_success"] is True
    assert result["temperature"] > 0
    assert result["calibrated_calibration_logloss"] <= result["raw_calibration_logloss"] + 1e-8
