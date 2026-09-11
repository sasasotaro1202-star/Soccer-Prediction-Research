import pytest

from src.legacy.v9_v12_adapter import LegacyEngineAdapter, make_pit_context


def test_adapter_rejects_unverified_context():
    adapter = LegacyEngineAdapter(lambda _: {"probabilities": {"H": 0.4, "D": 0.3, "A": 0.3}})
    with pytest.raises(ValueError, match="PIT-verified"):
        adapter.predict({"match_id": "m1", "prediction_cutoff_at_utc": "2026-01-01T12:00:00Z", "pit_verified": False})


def test_adapter_preserves_legacy_output_shape():
    def legacy(_):
        return {
            "probabilities": {"H": 0.5, "D": 0.2, "A": 0.3},
            "score_candidates": [{"home": 1, "away": 0, "probability": 0.2}],
            "metadata": {"model": "legacy-v9"},
        }

    adapter = LegacyEngineAdapter(legacy, source_version="V9")
    result = adapter.predict({"match_id": "m1", "prediction_cutoff_at_utc": "2026-01-01T12:00:00Z", "pit_verified": True})
    assert result.source_version == "V9"
    assert result.probabilities == {"H": 0.5, "D": 0.2, "A": 0.3}
    assert result.score_candidates[0]["home"] == 1


def test_make_context_does_not_invent_pit():
    with pytest.raises(ValueError):
        make_pit_context({"match_id": "m1", "prediction_cutoff_at_utc": "2026-01-01T12:00:00Z"})
