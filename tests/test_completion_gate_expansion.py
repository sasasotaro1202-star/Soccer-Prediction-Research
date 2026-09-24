from src.data.completion_gate import _canonical_source_for, _has_status


def test_expanded_catalog_has_safe_canonical_source_fallback():
    assert _canonical_source_for("UECL") == "UEFA"
    assert _canonical_source_for("UEFA_YOUTH_LEAGUE") == "UEFA"
    assert _canonical_source_for("UNKNOWN_FUTURE_TARGET") == "UNVERIFIED / no canonical source configured"


def test_status_counting_does_not_confuse_unavailable_with_available():
    assert _has_status("UNAVAILABLE", "UNAVAILABLE")
    assert not _has_status("UNAVAILABLE", "AVAILABLE")
    assert _has_status("AVAILABLE|UNAVAILABLE", "AVAILABLE")
    assert _has_status("AVAILABLE|UNAVAILABLE", "UNAVAILABLE")
