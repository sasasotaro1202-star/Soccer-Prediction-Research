from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from src.research.opta_like_source_probe import _probe, probe_all


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "soccer-source-probe.yml"


def test_probe_workflow_is_pinned_reproducible_and_fail_closed():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in text
    assert "group: soccer-source-probe" in text
    assert "cancel-in-progress: false" in text
    assert 'PYTHONHASHSEED: "0"' in text
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in text
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in text
    assert "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f" in text
    assert "set -euo pipefail" in text
    assert "@v4" not in text
    assert "@v5" not in text


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
