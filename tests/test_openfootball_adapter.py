import pandas as pd

from src.data.openfootball_adapter import parse_football_txt


def test_parse_football_txt_regular_match():
    raw = b"= Test Cup 2024/25\n  Tue Aug 13 2024\n    19:30  Home FC                v Away FC                 2-1 (1-0)\n"
    out = parse_football_txt(raw.decode(), "CAR", 2024, "https://example.invalid/test.txt", raw)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.home_team == "Home FC"
    assert row.away_team == "Away FC"
    assert row.home_goals == 2
    assert row.away_goals == 1
    assert row.result == "H"
    assert row.event_time_precision == "MINUTE"
    assert pd.isna(row.source_available_at_utc)


def test_penalty_shootout_is_draw_for_1x2_target():
    raw = b"= Test Cup 2024/25\n  Tue Aug 13 2024\n    19:30  Home FC                v Away FC                 4-3 pen. (1-1, 0-1)\n"
    out = parse_football_txt(raw.decode(), "CAR", 2024, "https://example.invalid/test.txt", raw)
    row = out.iloc[0]
    assert row.home_goals == 1
    assert row.away_goals == 1
    assert row.result == "D"


def test_no_publication_time_is_not_invented():
    raw = b"= Test Cup 2024/25\n  Tue Aug 13 2024\n    19:30  Home FC                v Away FC                 1-0\n"
    out = parse_football_txt(raw.decode(), "UCL", 2024, "https://example.invalid/test.txt", raw)
    assert pd.isna(out.iloc[0].source_available_at_utc)


def test_load_openfootball_season_refreshes_stale_cached_dates(tmp_path, monkeypatch):
    import src.data.openfootball_adapter as adapter

    stale = (
        b"= UEFA Champions League 2024/25\n"
        b"  Tue Aug 13 2026\n"
        b"    19:30  Home FC                v Away FC                 1-0\n"
    )
    fresh = (
        b"= UEFA Champions League 2024/25\n"
        b"  Tue Aug 13 2024\n"
        b"    19:30  Home FC                v Away FC                 1-0\n"
    )
    cache = tmp_path / "UCL_2024-25.txt"
    cache.write_bytes(stale)

    class Response:
        content = fresh
        def raise_for_status(self):
            return None

    calls = []
    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(adapter.requests, "get", fake_get)
    out = adapter.load_openfootball_season("UCL", 2024, cache_dir=str(tmp_path))

    assert len(calls) == 1
    assert len(out) == 1
    assert out.iloc[0]["kickoff_utc"] == pd.Timestamp("2024-08-13T19:30:00Z")
    assert cache.read_bytes() == fresh
    assert adapter._season_dates_valid(out, 2024)


def test_season_date_validator_rejects_calendar_outliers():
    import src.data.openfootball_adapter as adapter
    frame = pd.DataFrame({"kickoff_utc": [pd.Timestamp("2026-08-13T19:30:00Z")]})
    assert adapter._season_dates_valid(frame, 2024) is False
