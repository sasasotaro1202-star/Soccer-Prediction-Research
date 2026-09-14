import pandas as pd
import pytest

from src.prediction.prepare_fixtures import prepare_future_fixtures


def _history():
    rows = []
    base = pd.Timestamp("2026-01-01T12:00:00Z")
    pairs = [("A", "B", 2, 0), ("B", "A", 1, 1), ("A", "B", 1, 0), ("B", "A", 0, 2), ("A", "B", 2, 1)]
    for i, (home, away, hg, ag) in enumerate(pairs):
        kickoff = base + pd.Timedelta(days=i * 7)
        rows.append({
            "match_id": f"h{i}", "competition": "TEST", "season": 2026,
            "season_start": 2026, "kickoff_utc": kickoff,
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag,
            "source_available_at_utc": kickoff + pd.Timedelta(hours=2),
        })
    return pd.DataFrame(rows)


def _fixture():
    kickoff = pd.Timestamp("2026-02-10T12:00:00Z")
    return pd.DataFrame([{
        "match_id": "f1", "competition": "TEST", "season": 2026,
        "season_start": 2026, "kickoff_utc": kickoff,
        "home_team": "A", "away_team": "B",
        "source_available_at_utc": pd.Timestamp("2026-02-01T00:00:00Z"),
        "starter_status": "ANNOUNCED",
    }])


def test_prepare_future_fixture_reuses_pit_feature_builder():
    out = prepare_future_fixtures(_fixture(), _history())
    assert len(out) == 1
    assert out.loc[0, "match_id"] == "f1"
    assert bool(out.loc[0, "pit_verified"]) is True
    assert out.loc[0, "starter_status"] == "ANNOUNCED"
    assert pd.notna(out.loc[0, "home_elo"])
    assert "home_gf_3" in out.columns


def test_prepare_rejects_duplicate_fixture_identity():
    f = pd.concat([_fixture(), _fixture()], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate match_id"):
        prepare_future_fixtures(f, _history())


def test_prepare_rejects_unknown_kickoff():
    f = _fixture()
    f.loc[0, "kickoff_utc"] = "not-a-time"
    with pytest.raises(RuntimeError, match="invalid kickoff_utc"):
        prepare_future_fixtures(f, _history())
