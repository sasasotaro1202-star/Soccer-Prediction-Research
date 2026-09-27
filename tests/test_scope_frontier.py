from datetime import date
import json

from src.research.scope_frontier import discover


def test_frontier_never_auto_promotes_unknown_competition(monkeypatch, tmp_path):
    import src.research.scope_frontier as frontier

    monkeypatch.setattr(
        frontier,
        "_fetch_json",
        lambda day: {
            "events": [{
                "id": 1,
                "tournament": {
                    "uniqueTournament": {
                        "id": 9999,
                        "name": "Future Invitational Women",
                        "slug": "future-invitational-women",
                    }
                }
            }]
        },
    )
    output = tmp_path / "frontier.json"
    result = discover(2, str(output), start=date(2026, 9, 28))
    assert result["status"] == "OK"
    assert result["production_auto_promotion"] is False
    assert result["discovered_candidates"][0]["stage"] == "DISCOVERED"
    assert result["discovered_candidates"][0]["eligibility"] == "RESEARCH_ONLY_UNVERIFIED"
    assert json.loads(output.read_text())["discovered_candidates"]
