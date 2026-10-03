from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "chat-stream-resilience.md"

def test_chat_stream_resilience_is_async_and_durable():
    text = DOC.read_text(encoding="utf-8")
    assert "dispatch long-running work asynchronously" in text
    assert "never synchronously poll" in text
    assert "requests/state/<request_id>.json" in text
    assert "client is backgrounded" in text
    assert "ChatGPT application's UI streaming transport itself" in text

def test_chat_stream_resilience_preserves_fail_closed_rules():
    text = DOC.read_text(encoding="utf-8")
    assert "DISPATCHING" in text
    assert "must be reconciled rather than duplicated" in text
    assert "|| true" in text
    assert "PIT" in text
    assert "frozen holdout" in text
