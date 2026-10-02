from src.research.oos_window_signature import exact_oos_window_signature


def test_signature_is_deterministic_and_boundary_sensitive():
    rows = [
        {"fold": 1, "test_start": "2025-01-01T00:00:00+00:00", "test_end": "2025-01-31T00:00:00+00:00"},
        {"fold": 2, "test_start": "2025-02-01T00:00:00+00:00", "test_end": "2025-02-28T00:00:00+00:00"},
    ]
    a = exact_oos_window_signature(rows)
    b = exact_oos_window_signature(list(reversed(rows)))
    assert a == b
    assert a is not None
    assert a.startswith("oos:")
    assert len(a) == 20

    changed = exact_oos_window_signature([
        rows[0],
        {**rows[1], "test_end": "2025-02-27T00:00:00+00:00"},
    ])
    assert changed != a


def test_signature_fails_closed_on_missing_boundaries_or_fold():
    assert exact_oos_window_signature([
        {"fold": 1, "test_start": "2025-01-01T00:00:00+00:00"}
    ]) is None
    assert exact_oos_window_signature([
        {"fold": "not-a-fold", "test_start": "2025-01-01", "test_end": "2025-01-31"}
    ]) is None
    assert exact_oos_window_signature([]) is None


def test_signature_fails_closed_on_duplicate_fold_identity():
    rows = [
        {"fold": 1, "test_start": "2025-01-01", "test_end": "2025-01-31"},
        {"fold": 1, "test_start": "2025-02-01", "test_end": "2025-02-28"},
    ]
    assert exact_oos_window_signature(rows) is None


def test_signature_accepts_existing_oos_field_names():
    signature = exact_oos_window_signature([
        {
            "fold": 0,
            "oos_start": "2026-01-01T00:00:00+00:00",
            "oos_end": "2026-01-10T00:00:00+00:00",
        }
    ])
    assert signature is not None
