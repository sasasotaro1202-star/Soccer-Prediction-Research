import pytest

from src.research.xdata_pit_store import PITSnapshotStore, build_snapshot
from src.research.offline_evaluation import EvaluationRow, evaluate, compare_candidate_to_baseline


def test_pit_store_is_idempotent_and_cutoff_safe(tmp_path):
    store = PITSnapshotStore(tmp_path)
    snapshot = build_snapshot(
        source="clubelo", request_key="2026-09-14", entity_key="ARS",
        payload={"elo": 1900}, captured_at="2026-09-14T12:00:00Z",
        feature_available_at="2026-09-14T11:00:00Z",
        prediction_cutoff_at="2026-09-14T12:00:00Z",
    )
    assert store.put(snapshot) is True
    assert store.put(snapshot) is False
    rows = store.query_pit_safe(source="clubelo", entity_key="ARS", cutoff_at="2026-09-14T12:00:00Z")
    assert len(rows) == 1
    assert rows[0].pit_safe is True


def test_query_rejects_snapshot_recorded_for_future_cutoff(tmp_path):
    store = PITSnapshotStore(tmp_path)
    snapshot = build_snapshot(
        source="clubelo", request_key="future-cutoff", entity_key="ARS",
        payload={"elo": 1900}, captured_at="2026-09-14T12:00:00Z",
        feature_available_at="2026-09-14T11:00:00Z",
        prediction_cutoff_at="2026-09-15T12:00:00Z",
    )
    assert store.put(snapshot) is True
    assert store.query_pit_safe(source="clubelo", entity_key="ARS", cutoff_at="2026-09-14T12:00:00Z") == []


def test_missing_availability_is_not_pit_safe(tmp_path):
    store = PITSnapshotStore(tmp_path)
    snapshot = build_snapshot(
        source="statsbomb_open_data", request_key="m1", entity_key="ARS",
        payload={"xg": 1.2}, captured_at="2026-09-14T12:00:00Z",
    )
    assert snapshot.pit_safe is False
    assert store.put(snapshot) is True
    assert store.query_pit_safe(source="statsbomb_open_data", entity_key="ARS", cutoff_at="2026-09-14T12:00:00Z") == []


def test_offline_evaluation_rejects_non_pit_rows():
    rows = [
        EvaluationRow("2026-09-15T12:00:00Z", "2026-09-15T11:00:00Z", 0, (0.6, 0.2, 0.2), True, "s1"),
        EvaluationRow("2026-09-16T12:00:00Z", "2026-09-16T11:00:00Z", 1, (0.2, 0.5, 0.3), False, "s2"),
    ]
    result = evaluate(rows)
    assert result.n == 1
    assert result.rejected_rows == 1
    assert result.accuracy == 1.0


def test_candidate_comparison_only_returns_deltas():
    delta = compare_candidate_to_baseline(
        {"accuracy": .50, "log_loss": 1.02, "brier": .61},
        {"accuracy": .52, "log_loss": 1.00, "brier": .59},
    )
    assert delta["accuracy"] == pytest.approx(.02)
    assert delta["log_loss"] == pytest.approx(-.02)
    assert delta["brier"] == pytest.approx(-.02)
