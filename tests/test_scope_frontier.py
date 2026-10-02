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


def test_frontier_fetch_uses_resilient_http(monkeypatch):
    import src.research.scope_frontier as frontier

    observed = {}

    class Response:
        def json(self):
            return {"events": []}

    def fake_get(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        return Response()

    monkeypatch.setattr(frontier, "resilient_get", lambda getter, url, **kwargs: fake_get(url, **kwargs))
    payload = frontier._fetch_json(date(2026, 9, 28))
    assert payload == {"events": []}
    assert observed["timeout"] is None


def test_frontier_source_has_no_direct_urlopen():
    import inspect
    import src.research.scope_frontier as frontier

    source = inspect.getsource(frontier)
    assert "urlopen(" not in source
    assert "resilient_get(" in source
