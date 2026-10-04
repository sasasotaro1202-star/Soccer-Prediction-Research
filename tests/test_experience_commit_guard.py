from src import __path__  # noqa: F401
from scripts.experience_commit_guard import semantic_json_equal


def test_generated_timestamp_only_change_is_not_semantic():
    before = '{"status":"WARMUP","generated_at_utc":"2026-10-03T20:55:02Z"}'
    after = '{"status":"WARMUP","generated_at_utc":"2026-10-04T02:55:02Z"}'
    assert semantic_json_equal(before, after)


def test_nested_generated_timestamp_only_change_is_not_semantic():
    before = '{"status":"HOLD","summary":{"generated_at_utc":"a","rows":0}}'
    after = '{"status":"HOLD","summary":{"generated_at_utc":"b","rows":0}}'
    assert semantic_json_equal(before, after)


def test_real_status_change_is_semantic():
    before = '{"status":"WARMUP","generated_at_utc":"a"}'
    after = '{"status":"PROMOTION_CANDIDATE","generated_at_utc":"b"}'
    assert not semantic_json_equal(before, after)


def test_real_numeric_change_is_semantic():
    before = '{"rows":0,"generated_at_utc":"a"}'
    after = '{"rows":12,"generated_at_utc":"b"}'
    assert not semantic_json_equal(before, after)


def test_non_json_text_is_exact():
    assert semantic_json_equal("plain", "plain")
    assert not semantic_json_equal("plain", "changed")


def test_missing_optional_path_is_not_a_guard_failure(monkeypatch):
    from scripts import experience_commit_guard as mod

    monkeypatch.setattr(
        mod,
        "_run_git",
        lambda *args: None if args == ("show", ":optional-missing.json") else b"same",
    )
    assert mod.staged_semantic_change("optional-missing.json") is False
