import numpy as np

from src.research.dual_memory_oos import evaluate_dual_memory_oos, mix_probabilities


def _fold(start, n, rng):
    full = rng.dirichlet([4.0, 2.0, 2.0], size=n)
    recent = full.copy()
    shift = rng.random(n) < 0.35
    recent[shift] = rng.dirichlet([2.0, 4.0, 2.0], size=int(shift.sum()))
    return {"end": start, "te": start+n, "full": full, "recent": recent}


def test_mix_is_normalized_and_bounded():
    a = np.array([[0.8, 0.1, 0.1]])
    b = np.array([[0.1, 0.8, 0.1]])
    out = mix_probabilities(a, b, 0.6)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out > 0)


def test_dual_memory_is_chronological_research_only():
    rng = np.random.default_rng(7)
    folds = [_fold(i*60, 60, rng) for i in range(8)]
    y = np.concatenate([np.argmax(f["full"], axis=1) for f in folds])
    result = evaluate_dual_memory_oos(y, folds, min_blocks=6)
    assert result["status"] == "EVALUATED"
    assert result["samples"] == 480
    assert result["production_changed"] is False
    assert result["holdout_used_for_selection"] is False
    assert 0.20 <= result["final_recent_weight"] <= 0.80
