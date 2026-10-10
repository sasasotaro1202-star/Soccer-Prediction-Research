import json
from pathlib import Path

import src.research.scope_frontier as frontier


def test_catalogued_inactive_competition_is_research_only(monkeypatch, tmp_path):
    monkeypatch.setattr(frontier, "_fetch_json", lambda day: {
        "events": [{
            "tournament": {"uniqueTournament": {"name": "K League 1", "slug": "k-league-1", "id": 410}},
            "homeTeam": {"name": "A"},
            "awayTeam": {"name": "B"},
            "startTimestamp": 1790000000,
            "id": 123,
        }]
    })
    # Bypass the detailed event parser; discovery needs only tournament identity
    # and scheduled-event payload.
    out = tmp_path / "frontier.json"
    result = frontier.discover(1, str(out))
    assert result["catalogued_inactive_event_counts"]["KOR"] == 1
    item = next(x for x in result["discovered_candidates"] if x["id"] == "catalog:KOR")
    assert item["stage"] == "DISCOVERED"
    assert item["eligibility"] == "RESEARCH_ONLY_UNVERIFIED"
