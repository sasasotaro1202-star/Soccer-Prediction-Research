import numpy as np
import pandas as pd

from src.prediction.matchday_intelligence import apply_matchday_intelligence


def _fixtures(**extra):
    base = pd.DataFrame([{
        "kickoff_utc": "2026-09-25T12:00:00Z",
        "matchday_available_at_utc": "2026-09-25T08:00:00Z",
        "matchday_pit_verified": True,
        "matchday_source": "free-public-test-source",
    }])
    for key, value in extra.items():
        base[key] = value
    return base


def test_absent_matchday_layer_is_noop():
    fixtures = pd.DataFrame([{"kickoff_utc": "2026-09-25T12:00:00Z"}])
    base = np.asarray([[0.55, 0.25, 0.20]])
    out, diag = apply_matchday_intelligence(base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z"))
    assert np.allclose(out, base)
    assert diag.loc[0, "status"] == "ABSENT"


def test_valid_matchday_signals_are_bounded_and_normalized():
    fixtures = _fixtures(
        matchday_injury_impact_home=0.8,
        matchday_injury_impact_away=0.2,
        matchday_lineup_impact_home=0.6,
        matchday_lineup_impact_away=0.3,
        matchday_weather_penalty_home=0.2,
        matchday_weather_penalty_away=0.1,
        matchday_rest_diff_hours=12.0,
        matchday_market_p_home=0.50,
        matchday_market_p_draw=0.27,
        matchday_market_p_away=0.23,
    )
    base = np.asarray([[0.55, 0.25, 0.20]])
    out, diag = apply_matchday_intelligence(base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z"))
    assert diag.loc[0, "status"] == "APPLIED"
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)
    assert np.max(np.abs(out - base)) <= 0.0600001


def test_future_matchday_evidence_is_blocked():
    base = np.asarray([[0.55, 0.25, 0.20]])
    fixtures = _fixtures(matchday_available_at_utc="2026-09-25T10:00:00Z", matchday_injury_impact_home=0.9, matchday_injury_impact_away=0.1)
    out, diag = apply_matchday_intelligence(base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z"))
    assert np.allclose(out, base)
    assert diag.loc[0, "status"] == "BLOCKED_PIT_METADATA"


def test_invalid_market_is_blocked():
    base = np.asarray([[0.55, 0.25, 0.20]])
    fixtures = _fixtures(matchday_market_p_home=0.70, matchday_market_p_draw=0.25, matchday_market_p_away=0.20)
    out, diag = apply_matchday_intelligence(base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z"))
    assert np.allclose(out, base)
    assert diag.loc[0, "status"] == "BLOCKED_INVALID_SIGNAL"


def test_freshness_reduces_adjustment():
    base = np.asarray([[0.55, 0.25, 0.20]])
    fresh = _fixtures(matchday_injury_impact_home=0.9, matchday_injury_impact_away=0.1)
    stale = _fixtures(matchday_available_at_utc="2026-09-23T09:00:00Z", matchday_injury_impact_home=0.9, matchday_injury_impact_away=0.1)
    now = pd.Timestamp("2026-09-25T09:00:00Z")
    fresh_out, fresh_diag = apply_matchday_intelligence(base, fresh, now)
    stale_out, stale_diag = apply_matchday_intelligence(base, stale, now)
    assert fresh_diag.loc[0, "adjustment_l1"] > stale_diag.loc[0, "adjustment_l1"]
    assert fresh_diag.loc[0, "freshness"] > stale_diag.loc[0, "freshness"]


def test_raw_decimal_market_odds_are_devigged():
    fixtures = _fixtures(
        matchday_odds_home=2.00,
        matchday_odds_draw=3.60,
        matchday_odds_away=4.20,
    )
    base = np.asarray([[0.45, 0.30, 0.25]])
    out, diag = apply_matchday_intelligence(base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z"))
    assert diag.loc[0, "status"] == "APPLIED"
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)
    assert "market_odds" in diag.loc[0, "signal_names"]


def test_unconfirmed_lineup_signal_is_ignored():
    fixtures = _fixtures(
        starter_status="EXPECTED",
        matchday_lineup_impact_home=1.0,
        matchday_lineup_impact_away=0.0,
    )
    base = np.asarray([[0.55, 0.25, 0.20]])
    out, diag = apply_matchday_intelligence(base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z"))
    assert np.allclose(out, base)
    assert "lineup" not in diag.loc[0, "signal_names"]


def test_signal_confidence_scales_matchday_adjustment():
    high = _fixtures(
        matchday_signal_confidence=1.0,
        matchday_injury_impact_home=0.9,
        matchday_injury_impact_away=0.1,
    )
    low = _fixtures(
        matchday_signal_confidence=0.25,
        matchday_injury_impact_home=0.9,
        matchday_injury_impact_away=0.1,
    )
    base = np.asarray([[0.55, 0.25, 0.20]])
    now = pd.Timestamp("2026-09-25T09:00:00Z")
    high_out, _ = apply_matchday_intelligence(base, high, now)
    low_out, _ = apply_matchday_intelligence(base, low, now)
    assert np.abs(high_out - base).sum() > np.abs(low_out - base).sum()


def test_nan_optional_groups_do_not_block_other_matchday_signals():
    fixtures = _fixtures(
        matchday_injury_impact_home=0.8,
        matchday_injury_impact_away=0.2,
        matchday_rest_diff_hours=np.nan,
        matchday_market_p_home=np.nan,
        matchday_market_p_draw=np.nan,
        matchday_market_p_away=np.nan,
        matchday_odds_home=np.nan,
        matchday_odds_draw=np.nan,
        matchday_odds_away=np.nan,
    )
    base = np.asarray([[0.55, 0.25, 0.20]])
    out, diag = apply_matchday_intelligence(
        base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z")
    )
    assert diag.loc[0, "status"] == "APPLIED"
    assert "injury" in diag.loc[0, "signal_names"]


def test_market_probability_prevents_duplicate_odds_diagnostic():
    fixtures = _fixtures(
        matchday_market_p_home=0.50,
        matchday_market_p_draw=0.27,
        matchday_market_p_away=0.23,
        matchday_odds_home=2.00,
        matchday_odds_draw=3.60,
        matchday_odds_away=4.20,
    )
    base = np.asarray([[0.50, 0.27, 0.23]])
    _, diag = apply_matchday_intelligence(
        base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z")
    )
    assert diag.loc[0, "signal_names"] == "market"
    assert int(diag.loc[0, "signal_count"]) == 3



def test_severe_weather_flattens_probabilities_without_creating_directional_bias():
    fixtures = _fixtures(
        matchday_weather_severity=1.0,
        matchday_weather_penalty_home=1.0,
        matchday_weather_penalty_away=1.0,
    )
    base = np.asarray([[0.80, 0.12, 0.08]])
    out, diag = apply_matchday_intelligence(
        base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z")
    )
    assert diag.loc[0, "status"] == "APPLIED"
    assert "weather_uncertainty" in diag.loc[0, "signal_names"]
    assert out[0].max() < base[0].max()
    assert out[0, 1] > base[0, 1]
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_large_model_market_disagreement_is_treated_as_uncertainty():
    fixtures = _fixtures(
        matchday_market_p_home=0.20,
        matchday_market_p_draw=0.30,
        matchday_market_p_away=0.50,
    )
    base = np.asarray([[0.80, 0.12, 0.08]])
    out, diag = apply_matchday_intelligence(
        base, fixtures, pd.Timestamp("2026-09-25T09:00:00Z")
    )
    assert diag.loc[0, "status"] == "APPLIED"
    assert "market_uncertainty" in diag.loc[0, "signal_names"]
    assert out[0].max() < base[0].max()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)
