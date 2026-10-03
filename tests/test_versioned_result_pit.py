from __future__ import annotations

import pandas as pd
import pytest

from src.data import versioned_result_pit as mod


def _history():
    return pd.DataFrame([
        {
            "match_id": "m1",
            "competition": "EPL",
            "season_start": 2024,
            "kickoff_utc": "2024-08-17T15:00:00Z",
            "kickoff_time_available": True,
            "home_team": "Arsenal",
            "away_team": "Wolves",
            "home_goals": 2.0,
            "away_goals": 0.0,
        },
        {
            "match_id": "m2",
            "competition": "EPL",
            "season_start": 2024,
            "kickoff_utc": "2024-08-18T15:00:00Z",
            "kickoff_time_available": True,
            "home_team": "Chelsea",
            "away_team": "Manchester City",
            "home_goals": 0.0,
            "away_goals": 2.0,
        },
    ])


def _snapshot():
    return {
        "matches": [
            {
                "date": "2024-08-17",
                "team1": "Arsenal FC",
                "team2": "Wolverhampton Wanderers FC",
                "score": {"ft": [2, 0]},
            },
            {
                "date": "2024-08-18",
                "team1": "Chelsea FC",
                "team2": "Manchester City FC",
                "score": {"ft": [0, 2]},
            },
        ]
    }


def test_versioned_bridge_requires_conservative_post_match_bound():
    kickoff = pd.Timestamp("2024-08-17T15:00:00Z")
    row = pd.Series({
        "kickoff_utc": kickoff,
        "kickoff_time_available": True,
    })
    bound, reason = mod._lower_bound(row)
    assert bound == kickoff + pd.Timedelta(minutes=180)
    assert reason == "kickoff_plus_180m"


def test_versioned_bridge_rejects_unknown_competitions():
    with pytest.raises(ValueError, match="unsupported competition"):
        mod.season_path("UCL", 2024)


def test_versioned_bridge_verifies_exact_result_once(monkeypatch, tmp_path):
    history = _history()
    commits = [
        {
            "sha": "early",
            "commit": {"committer": {"date": "2024-08-17T16:00:00Z"}},
        },
        {
            "sha": "late",
            "commit": {"committer": {"date": "2024-08-17T19:30:00Z"}},
        },
    ]

    def fake_commits(path, *, cache_dir, timeout):
        return commits

    def fake_file(path, sha, *, cache_dir, timeout):
        return _snapshot()

    monkeypatch.setattr(mod, "_commits", fake_commits)
    monkeypatch.setattr(mod, "_file_at_commit", fake_file)

    enriched, report = mod.apply_bulk(
        history,
        max_groups=1,
        rows_per_group=2,
        cache_dir=tmp_path,
    )

    assert enriched.loc[0, "versioned_result_evidence_status"] == "VERIFIED"
    assert enriched.loc[0, "versioned_result_commit_sha"] == "late"
    assert enriched.loc[0, "versioned_result_available_at_utc"] == "2024-08-17T19:30:00+00:00"
    assert int(report.loc[0, "verified_rows"]) == 1


def test_versioned_bridge_does_not_accept_ambiguous_snapshot_keys(monkeypatch, tmp_path):
    history = _history().iloc[[0]].copy()
    commits = [{
        "sha": "late",
        "commit": {"committer": {"date": "2024-08-17T19:30:00Z"}},
    }]
    ambiguous = {
        "matches": [
            {
                "date": "2024-08-17",
                "team1": "Arsenal FC",
                "team2": "Wolverhampton Wanderers FC",
                "score": {"ft": [2, 0]},
            },
            {
                "date": "2024-08-17",
                "team1": "Arsenal",
                "team2": "Wolverhampton Wanderers",
                "score": {"ft": [2, 0]},
            },
        ]
    }
    monkeypatch.setattr(mod, "_commits", lambda *args, **kwargs: commits)
    monkeypatch.setattr(mod, "_file_at_commit", lambda *args, **kwargs: ambiguous)

    enriched, report = mod.apply_bulk(
        history, max_groups=1, rows_per_group=1, cache_dir=tmp_path
    )

    assert enriched.loc[0, "versioned_result_evidence_status"] == "UNVERIFIABLE"
    assert int(report.loc[0, "ambiguous_match_keys"]) == 1


def test_versioned_bridge_is_bounded_by_group_limit(monkeypatch, tmp_path):
    history = pd.concat([
        _history(),
        _history().assign(
            match_id=["m3", "m4"],
            season_start=2023,
        ),
    ], ignore_index=True)
    seen = []

    monkeypatch.setattr(
        mod,
        "_commits",
        lambda path, **kwargs: seen.append(path) or [],
    )

    _, report = mod.apply_bulk(history, max_groups=1, cache_dir=tmp_path)
    assert len(report) == 1
    assert len(seen) == 1


def test_resume_skips_completed_groups_and_advances_to_next_group(monkeypatch, tmp_path):
    history = pd.concat([
        _history(),
        _history().assign(
            match_id=["m3", "m4"],
            season_start=2023,
        ),
    ], ignore_index=True)
    history["versioned_result_evidence_status"] = "UNVERIFIABLE"
    history.loc[history["season_start"] == 2024, "versioned_result_evidence_status"] = "VERIFIED"

    seen = []
    monkeypatch.setattr(
        mod,
        "_commits",
        lambda path, **kwargs: seen.append(path) or [],
    )

    _, report = mod.apply_bulk(history, max_groups=1, cache_dir=tmp_path)
    assert len(report) == 1
    assert report.iloc[0]["season_start"] == 2023
    assert seen == ["2023-24/en.1.json"]



def test_versioned_input_selection_rejects_legacy_resume_schema(tmp_path):
    primary = tmp_path / "normalized.csv"
    resume = tmp_path / "resume.csv"
    primary.write_text("competition,season_start,kickoff_utc,home_team,away_team,home_goals,away_goals\n", encoding="utf-8")
    resume.write_text("legacy_column\nlegacy\n", encoding="utf-8")

    path, reason = mod.select_input_path(primary, resume)

    assert path == primary
    assert reason == "RESUME_SCHEMA_INVALID"


def test_versioned_input_selection_accepts_usable_resume_schema(tmp_path):
    primary = tmp_path / "normalized.csv"
    resume = tmp_path / "resume.csv"
    primary.write_text("competition,season_start,kickoff_utc,home_team,away_team,home_goals,away_goals\n", encoding="utf-8")
    resume.write_text(
        "competition,season_start,kickoff_utc,home_team,away_team,home_goals,away_goals,versioned_result_evidence_status\n"
        "EPL,2024,2024-08-17T15:00:00Z,Arsenal,Wolves,2,0,UNVERIFIABLE\n",
        encoding="utf-8",
    )

    path, reason = mod.select_input_path(primary, resume)

    assert path == resume
    assert reason == "RESUME_VALID"


def test_empty_input_schema_is_preserved(tmp_path, monkeypatch):
    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "evidence.csv"
    report_path = tmp_path / "report.csv"
    input_path.write_bytes(b"")

    monkeypatch.setattr(
        mod,
        "apply_bulk",
        lambda history, **kwargs: (
            history.assign(**{
                "versioned_result_source": "openfootball/football.json",
                "versioned_result_available_at_utc": pd.NaT,
                "versioned_result_evidence_status": "UNVERIFIABLE",
                "versioned_result_commit_sha": None,
                "versioned_result_evidence_url": None,
                "versioned_result_evidence_reason": "empty_input",
            }),
            pd.DataFrame(columns=mod.REPORT_COLUMNS),
        ),
    )
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "versioned_result_pit",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
    )

    assert mod.main() == 0
    evidence = pd.read_csv(output_path)
    assert "versioned_result_evidence_status" in evidence.columns
    assert len(evidence) == 0
    report = pd.read_csv(report_path)
    assert list(report.columns) == list(mod.REPORT_COLUMNS)

def test_versioned_bridge_paginates_commit_history(monkeypatch, tmp_path):
    first_page = [
        {"sha": "newest", "commit": {"committer": {"date": "2026-01-01T00:00:00Z"}}}
    ] + [
        {"sha": f"filler-{i}", "commit": {"committer": {"date": "2026-01-01T00:00:00Z"}}}
        for i in range(mod.DEFAULT_PER_PAGE - 1)
    ]
    pages = {
        1: first_page,
        2: [{"sha": "older", "commit": {"committer": {"date": "2025-01-01T00:00:00Z"}}}],
    }
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_request(url, *, params=None, timeout=None, retries=6):
        calls.append(dict(params or {}))
        return FakeResponse(pages.get(int((params or {}).get("page", 1)), []))

    monkeypatch.setattr(mod, "_request", fake_request)

    commits = mod._commits("2025-26/en.1.json", cache_dir=tmp_path, max_pages=12)

    assert len(commits) == mod.DEFAULT_PER_PAGE + 1
    assert commits[0]["sha"] == "newest"
    assert commits[-1]["sha"] == "older"
    assert [c["page"] for c in calls] == [1, 2]
    assert all(c["per_page"] == mod.DEFAULT_PER_PAGE for c in calls)


def test_versioned_bridge_respects_bounded_commit_page_limit(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        def json(self):
            return [{"sha": "page", "commit": {"committer": {"date": "2026-01-01T00:00:00Z"}}}] * mod.DEFAULT_PER_PAGE

    def fake_request(url, *, params=None, timeout=None, retries=6):
        calls.append(int((params or {}).get("page", 0)))
        return FakeResponse()

    monkeypatch.setattr(mod, "_request", fake_request)

    commits = mod._commits("2025-26/en.1.json", cache_dir=tmp_path, max_pages=3)

    assert len(commits) == mod.DEFAULT_PER_PAGE * 3
    assert calls == [1, 2, 3]

