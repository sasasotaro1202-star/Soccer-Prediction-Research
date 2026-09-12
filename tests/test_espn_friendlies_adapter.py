import pandas as pd

from src.data.espn_friendlies_adapter import _event_rows, _season_start


def test_friendlies_use_european_style_season():
    assert _season_start(pd.Timestamp("2025-08-01T12:00:00Z")) == 2025
    assert _season_start(pd.Timestamp("2026-02-01T12:00:00Z")) == 2025


def test_event_parser_preserves_team_ids_and_penalty_draw():
    payload = {
        "events": [{
            "id": "123",
            "date": "2025-08-01T12:00:00Z",
            "competitions": [{
                "status": {"type": {"name": "STATUS_FINAL_PEN"}},
                "competitors": [
                    {"homeAway": "home", "score": "1", "team": {"id": "10", "displayName": "Alpha"}},
                    {"homeAway": "away", "score": "1", "team": {"id": "20", "displayName": "Beta"}},
                ],
            }],
        }]
    }
    out = _event_rows(payload, "hash", pd.Timestamp("2026-01-01T00:00:00Z"))
    assert len(out) == 1
    assert out[0]["result"] == "D"
    assert out[0]["home_team_id"] == "10"
    assert out[0]["away_team_id"] == "20"
    assert out[0]["season"] == "2025/26"
