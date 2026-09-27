import numpy as np

from src.research.case_risk_oos import evaluate_case_risk_oof


def _make_fold(start, n, rng):
    # Two base models with complementary confidence; the fold target is tied to
    # a synthetic confidence regime only to make the risk metric deterministic.
    p = rng.dirichlet([2.0, 2.0, 2.0], size=n)
    q = p.copy()
    low_conf = np.max(p, axis=1) < 0.5
    q[low_conf] = rng.dirichlet([1.0, 1.0, 1.0], size=int(low_conf.sum()))
    y = np.argmax(p, axis=1)
    if n:
        # Inject deterministic errors concentrated in low-confidence rows.
        idx = np.flatnonzero(low_conf)
        y[idx] = (y[idx] + 1) % 3
    return {"end": start, "te": start + n, "preds": {"m1": p, "m2": q}, "context": np.max(p, axis=1)[:, None]}


def test_case_risk_oos_is_chronological_and_reports_coverage():
    rng = np.random.default_rng(42)
    folds = [_make_fold(i * 80, 80, rng) for i in range(6)]
    y = np.concatenate([
        np.argmax(f["preds"]["m1"], axis=1) for f in folds
    ])
    # Rebuild y exactly as each fold's injected error target.
    for f in folds:
        idx = np.flatnonzero(np.max(f["preds"]["m1"], axis=1) < 0.5)
        local = y[f["end"]:f["te"]]
        local[idx] = (local[idx] + 1) % 3
        y[f["end"]:f["te"]] = local

    result = evaluate_case_risk_oof(y, ("m1", "m2"), folds, min_training_rows=120)
    assert result["status"] == "EVALUATED"
    assert result["oos_rows"] == 480
    assert result["aurc"] is not None
    assert "0.80" in result["risk_coverage"]
    assert all("baseline_confidence_auc" in fold for fold in result["fold_metrics"])
