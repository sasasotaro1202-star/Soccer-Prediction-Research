from __future__ import annotations

import pandas as pd

from src.data import openfootball_international_adapter as mod


def test_international_path_patterns_cover_major_project_targets():
    samples = {
        "UEFA_EURO_M": "uefa_euro/2024_uefa_euro.txt",
        "UEFA_EURO_QUALI_M": "uefa_euro_qualification/2024_uefa_euro_qualification.txt",
        "UEFA_NATIONS_LEAGUE_M": "uefa_nations_league/2025_uefa_nations_league.txt",
        "ASIAN_CUP": "afc_asian_cup/2024_afc_asian_cup.txt",
        "WORLD_CUP": "fifa_world_cup/2026_fifa_world_cup.txt",
        "WORLD_CUP_QUALI": "fifa_world_cup_qualification/2026_fifa_world_cup_qualification.txt",
        "INTL_M": "friendly/2026_friendly.txt",
    }
    for competition, path in samples.items():
        assert mod.PATH_PATTERNS[competition].match(path)


def test_world_and_asian_cup_use_calendar_year_season_labels(monkeypatch):
    paths = [
        "fifa_world_cup/2022_fifa_world_cup.txt",
        "afc_asian_cup/2024_afc_asian_cup.txt",
        "friendly/2024_friendly.txt",
    ]
    raw = """= Tournament\nSun Jan 1\nA 1-0 B @ Venue\n"""
    monkeypatch.setattr(mod, "_tree_paths", lambda: paths)
    monkeypatch.setattr(mod, "_fetch_text", lambda p, c: (raw, raw.encode(), mod.RAW_BASE + p))
    history, _ = mod.load_openfootball_international_history(start_year=2022, end_year=2024, max_workers=2)
    assert not history.empty
    assert set(history.loc[history["competition"] == "WORLD_CUP", "season"]) == {"2022"}
    assert set(history.loc[history["competition"] == "ASIAN_CUP", "season"]) == {"2024"}
    assert set(history.loc[history["competition"] == "INTL_M", "season"]) == {"2024"}
    intl_kickoff = history.loc[history["competition"] == "INTL_M", "kickoff_utc"].iloc[0]
    assert intl_kickoff.year == 2024
    asian_kickoff = history.loc[history["competition"] == "ASIAN_CUP", "kickoff_utc"].iloc[0]
    assert asian_kickoff.year == 2024


def test_international_loader_uses_discovered_paths_without_network_for_download(tmp_path, monkeypatch):
    paths = [
        "uefa_euro/2024_uefa_euro.txt",
        "uefa_nations_league/2025_uefa_nations_league.txt",
        "fifa_world_cup/2026_fifa_world_cup.txt",
        "ignored/readme.md",
    ]
    raw_text = """= UEFA Euro 2024
Sat Jun 15
Germany v Scotland 5-1
"""
    monkeypatch.setattr(mod, "_tree_paths", lambda: paths)

    def fake_fetch(relative_path, cache_dir):
        return raw_text, raw_text.encode("utf-8"), mod.RAW_BASE + relative_path

    monkeypatch.setattr(mod, "_fetch_text", fake_fetch)
    history, coverage = mod.load_openfootball_international_history(
        start_year=2024,
        end_year=2024,
        max_workers=2,
    )

    assert not history.empty
    assert set(history["competition"]) == {"UEFA_EURO_M"}
    assert int(history["home_goals"].iloc[0]) == 5
    assert int(history["away_goals"].iloc[0]) == 1
    assert coverage["status"].tolist() == ["AVAILABLE"]
    assert coverage["competition"].tolist() == ["UEFA_EURO_M"]


def test_cached_tree_paths_returns_only_nonempty_txt_files(tmp_path):
    import src.data.openfootball_international_adapter as adapter

    good = tmp_path / "friendly" / "2025_friendly.txt"
    good.parent.mkdir(parents=True)
    good.write_text("sample", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    ignored = tmp_path / "notes.md"
    ignored.write_text("sample", encoding="utf-8")

    assert adapter._cached_tree_paths(str(tmp_path)) == ["friendly/2025_friendly.txt"]


def test_international_history_degrades_when_tree_discovery_is_unavailable(tmp_path, monkeypatch):
    import src.data.openfootball_international_adapter as adapter

    monkeypatch.setattr(adapter, "_tree_paths", lambda: (_ for _ in ()).throw(RuntimeError("GitHub 403")))
    history, coverage = adapter.load_openfootball_international_history(
        start_year=2025,
        end_year=2025,
        max_workers=1,
        cache_dir=str(tmp_path),
    )

    assert history.empty
    assert coverage.empty


def test_international_history_uses_cached_paths_when_tree_discovery_fails(tmp_path, monkeypatch):
    import src.data.openfootball_international_adapter as adapter

    cached = tmp_path / "friendly" / "2025_friendly.txt"
    cached.parent.mkdir(parents=True)
    cached.write_text(
        "Sat Jan 4 2025\n15:00 Alpha v Beta 1-0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "_tree_paths", lambda: (_ for _ in ()).throw(RuntimeError("GitHub 403")))
    history, coverage = adapter.load_openfootball_international_history(
        start_year=2025,
        end_year=2025,
        max_workers=1,
        cache_dir=str(tmp_path),
    )

    assert not history.empty
    assert coverage["discovery_mode"].eq("CACHE_FALLBACK").all()
