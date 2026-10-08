import pandas as pd
import pytest

statsmodels = pytest.importorskip("statsmodels")

from src.research.external_oss.statsmodels_poisson import (
    fit_statsmodels_poisson_score_model,
    predict_statsmodels_poisson_distribution,
)


def _history() -> pd.DataFrame:
    rows = []
    fixtures = [
        ("A", "B", 2, 0),
        ("B", "A", 1, 1),
        ("A", "C", 1, 0),
        ("C", "A", 0, 2),
        ("B", "C", 0, 1),
        ("C", "B", 2, 1),
        ("A", "B", 1, 2),
        ("B", "A", 2, 0),
        ("A", "C", 0, 0),
        ("C", "A", 1, 1),
        ("B", "C", 2, 2),
        ("C", "B", 0, 1),
    ]
    for i, (home, away, hg, ag) in enumerate(fixtures):
        rows.append(
            {
                "match_id": f"m{i}",
                "kickoff_utc": f"2026-01-{i + 1:02d}T12:00:00Z",
                "source_available_at_utc": f"2026-01-{i + 1:02d}T13:00:00Z",
                "home_team": home,
                "away_team": away,
                "competition": "TEST",
                "home_goals": hg,
                "away_goals": ag,
                "pit_verified": True,
            }
        )
    return pd.DataFrame(rows)


def test_fit_and_distribution_are_finite_and_normalized():
    model = fit_statsmodels_poisson_score_model(
        _history(), prediction_cutoff_utc="2026-02-01T00:00:00Z"
    )
    distribution = predict_statsmodels_poisson_distribution(
        model, "A", "B", "TEST", max_goals=8
    )
    assert model.training_rows == 12
    assert len(distribution) == 81
    assert all(h >= 0 and a >= 0 for h, a, _ in distribution)
    assert all(float(p) >= 0.0 for _, _, p in distribution)
    assert sum(float(p) for _, _, p in distribution) == pytest.approx(1.0, abs=1e-9)


def test_unknown_team_fails_closed():
    model = fit_statsmodels_poisson_score_model(
        _history(), prediction_cutoff_utc="2026-02-01T00:00:00Z"
    )
    with pytest.raises(RuntimeError, match="lacks PIT-trained"):
        predict_statsmodels_poisson_distribution(
            model, "UNKNOWN", "B", "TEST", max_goals=8
        )


def test_unverified_rows_do_not_enter_training():
    history = _history()
    history.loc[0, "pit_verified"] = False
    model = fit_statsmodels_poisson_score_model(
        history, prediction_cutoff_utc="2026-02-01T00:00:00Z"
    )
    assert model.training_rows == 11


def test_cutoff_and_unknown_timestamps_fail_closed():
    history = _history()
    history.loc[0, "source_available_at_utc"] = "2026-02-02T00:00:00Z"
    model = fit_statsmodels_poisson_score_model(
        history, prediction_cutoff_utc="2026-02-01T00:00:00Z"
    )
    assert model.training_rows == 11

    broken = _history()
    broken.loc[0, "source_available_at_utc"] = None
    model = fit_statsmodels_poisson_score_model(
        broken, prediction_cutoff_utc="2026-02-01T00:00:00Z"
    )
    assert model.training_rows == 11
