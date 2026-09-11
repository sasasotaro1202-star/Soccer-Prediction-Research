from __future__ import annotations


def low_high_result(home_goals: int, away_goals: int, threshold: float = 2.5) -> int:
    """1=High, 0=Low. Threshold is fixed before evaluation."""
    return int((home_goals + away_goals) > threshold)


def low_high_metrics(actual, predicted_probability, threshold: float = 2.5):
    # Probability is P(High); classification is intentionally secondary to proper scoring.
    y = [low_high_result(int(h), int(a), threshold) for h, a in actual]
    pred = [int(p >= 0.5) for p in predicted_probability]
    accuracy = sum(a == b for a, b in zip(y, pred)) / len(y) if y else float("nan")
    return {"threshold": threshold, "accuracy": accuracy, "n": len(y)}
