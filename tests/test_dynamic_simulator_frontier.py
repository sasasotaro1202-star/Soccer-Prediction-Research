from pathlib import Path

from src.research.dynamic_simulator_frontier import inspect


def test_frontier_is_research_only():
    status = inspect(Path("."))
    assert status["research_only"] is True
    assert status["production_changed"] is False
    assert status["promotion_candidate"] is False
    assert status["performance_verified"] is False
    assert status["frozen_holdout_access_allowed"] is False
    assert status["pit_required"] is True
    assert status["capabilities"]["pit_contract"] is True
    assert status["capabilities"]["score_distribution"] is True
    assert status["capabilities"]["oos"] is True


def test_dynamic_hazard_is_optional_frontier_capability():
    status = inspect(Path("."))
    assert isinstance(status["capabilities"]["dynamic_hazard_candidate"], bool)
    assert status["status"] in {"DYNAMIC_MATCH_STATE_WAITING", "DYNAMIC_MATCH_STATE_READY"}
