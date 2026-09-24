from pathlib import Path


def test_9h_queue_watchdog_does_not_dispatch_on_main_push():
    text = Path(".github/workflows/soccer-9h-queue-watchdog.yml").read_text(encoding="utf-8")
    on_block = text.split("on:", 1)[1].split("permissions:", 1)[0]
    assert "\n  push:\n" not in on_block
    assert 'cron: "*/5 * * * *"' in on_block
    assert "workflow_dispatch:" in on_block
