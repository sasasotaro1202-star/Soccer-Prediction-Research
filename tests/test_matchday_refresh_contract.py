from pathlib import Path


def test_matchday_refresh_contract_is_read_only_and_quarter_hourly():
    content = Path(".github/workflows/soccer-matchday-intelligence.yml").read_text(encoding="utf-8")
    assert 'cron: "7,22,37,52 * * * *"' in content
    assert "contents: read" in content
    assert "cancel-in-progress: true" in content
    assert "upload-artifact@v6" in content
