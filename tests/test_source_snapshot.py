from src.data.source_snapshot import append_snapshot, make_snapshot, pit_eligible, read_snapshots


def test_snapshot_is_eligible_only_when_available_before_cutoff():
    s = make_snapshot(
        source_key="statsbomb_open",
        event_id="m1",
        entity_id="team_a",
        payload={"xg": 1.2},
        available_at="2026-01-01T10:00:00+00:00",
        retrieved_at="2026-01-02T00:00:00+00:00",
        prediction_cutoff="2026-01-01T12:00:00+00:00",
    )
    assert pit_eligible(s)


def test_late_snapshot_is_stored_but_not_pit_eligible():
    s = make_snapshot(
        source_key="skillcorner_open",
        event_id="m1",
        entity_id="team_a",
        payload={"speed": 5.0},
        available_at="2026-01-01T13:00:00+00:00",
        retrieved_at="2026-01-02T00:00:00+00:00",
        prediction_cutoff="2026-01-01T12:00:00+00:00",
    )
    assert not pit_eligible(s)


def test_snapshot_append_is_immutable_and_preserves_history(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    a = make_snapshot(
        source_key="understat",
        event_id="m1",
        entity_id="shot1",
        payload={"xg": 0.2},
        available_at="2026-01-01T10:00:00+00:00",
        prediction_cutoff="2026-01-01T11:00:00+00:00",
        retrieved_at="2026-01-01T12:00:00+00:00",
    )
    b = make_snapshot(
        source_key="understat",
        event_id="m1",
        entity_id="shot1",
        payload={"xg": 0.3},
        available_at="2026-01-01T12:00:00+00:00",
        prediction_cutoff="2026-01-01T11:00:00+00:00",
        retrieved_at="2026-01-01T12:30:00+00:00",
    )
    append_snapshot(a, path)
    append_snapshot(b, path)
    rows = read_snapshots(path)
    assert len(rows) == 2
    assert rows[0].payload_hash != rows[1].payload_hash
