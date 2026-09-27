from unittest.mock import patch
from urllib.error import HTTPError

from src.research.opta_like_source_probe import _probe, probe_all


def test_probe_falls_back_from_head_405_to_ranged_get():
    class Headers:
        def get(self, key, default=""):
            return "text/plain"

    class Response:
        status = 206
        headers = Headers()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def geturl(self):
            return "https://example.test/final"

    with patch("src.research.opta_like_source_probe.urlopen", side_effect=[
        HTTPError("https://example.test", 405, "method not allowed", {}, None),
        Response(),
    ]):
        result = _probe("https://example.test")

    assert result["status"] == "REACHABLE"
    assert result["method"] == "GET_RANGE"
    assert result["http_status"] == 206


def test_probe_all_preserves_safety_metadata():
    with patch("src.research.opta_like_source_probe._probe", return_value={"status":"REACHABLE","http_status":200}):
        result = probe_all()
    assert result["total"] >= 20
    assert result["reachable"] == result["total"]
    assert all(x["pit_status"] for x in result["results"])
    assert all(x["license_status"] for x in result["results"])
