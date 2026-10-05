from pathlib import Path


def test_integrity_audit_uses_timezone_aware_liveness_calculation() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    assert "from datetime import datetime, timezone" in workflow
    assert 'datetime.fromisoformat(raw.replace("Z", "+00:00"))' in workflow
    assert "created_at.astimezone(timezone.utc).timestamp()" in workflow
    assert 'time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%SZ"))' not in workflow


def test_integrity_audit_remains_read_only_and_fail_closed() -> None:
    workflow = Path(".github/workflows/soccer-automation-integrity-audit.yml").read_text(
        encoding="utf-8"
    )
    assert "permissions:" in workflow
    assert "actions: read" in workflow
    assert "contents: read" in workflow
    assert "CONTROL-PLANE INTEGRITY AUDIT HOLD/FAIL" in workflow
    assert '"production_change_allowed": False' in workflow
    assert '"performance_claim_allowed": False' in workflow
    assert '"frozen_holdout_access_allowed": False' in workflow
