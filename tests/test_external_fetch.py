from datetime import datetime, timezone

from src.data.external_fetch import content_sha256, pit_is_safe, request_key


def test_request_key_is_deterministic_and_order_independent():
    a = request_key("https://example.test/data", {"b": 2, "a": 1})
    b = request_key("https://example.test/data", {"a": 1, "b": 2})
    assert a == b
    assert len(a) == 64


def test_content_hash_is_sha256():
    assert content_sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_pit_gate_requires_real_timestamps_and_cutoff_order():
    assert pit_is_safe("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z")
    assert not pit_is_safe("2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z")
    assert not pit_is_safe(None, "2026-01-01T00:00:01Z")
    assert not pit_is_safe("2026-01-01T00:00:00", "2026-01-01T00:00:01Z")


def test_iso_timestamp_comparison_handles_offsets():
    assert pit_is_safe("2026-01-01T09:00:00+09:00", "2026-01-01T00:00:01Z")
