import pandas as pd

from src.data.competition_sources import TARGET_COMPETITIONS
from src.prediction.runner import _eligible_fixtures


def _fixture(competition: str) -> dict:
    return {
        "match_id": f"m-{competition}",
        "kickoff_utc": "2026-09-25T20:00:00Z",
        "home_team": "Home",
        "away_team": "Away",
        "competition": competition,
        "source_available_at_utc": "2026-09-25T10:00:00Z",
        "pit_verified": True,
        "starter_status": "EXPECTED",
    }


def test_target_competition_is_production_eligible():
    frame = pd.DataFrame([_fixture("EPL")])
    out = _eligible_fixtures(frame, pd.Timestamp("2026-09-25T12:00:00Z"))
    assert len(out) == 1
    assert out.iloc[0]["competition"] == "EPL"


def test_auxiliary_competition_is_blocked_from_production_prediction():
    frame = pd.DataFrame([_fixture("MLS"), _fixture("EPL")])
    out = _eligible_fixtures(frame, pd.Timestamp("2026-09-25T12:00:00Z"))
    assert out["competition"].tolist() == ["EPL"]
    assert "MLS" not in set(out["competition"])


def test_every_active_target_competition_survives_scope_filter():
    frame = pd.DataFrame([_fixture(comp) for comp in TARGET_COMPETITIONS])
    out = _eligible_fixtures(frame, pd.Timestamp("2026-09-25T12:00:00Z"))
    assert set(out["competition"]) == set(TARGET_COMPETITIONS)
