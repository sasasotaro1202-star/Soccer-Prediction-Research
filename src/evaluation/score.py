from __future__ import annotations

import math
import numpy as np


def score_distribution(home_lambda: float, away_lambda: float, max_goals: int = 7):
    cells = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ph = math.exp(-home_lambda) * home_lambda**h / math.factorial(h)
            pa = math.exp(-away_lambda) * away_lambda**a / math.factorial(a)
            cells.append((h, a, ph * pa))
    z = sum(p for _, _, p in cells) or 1.0
    return [(h, a, p / z) for h, a, p in sorted(cells, key=lambda x: x[2], reverse=True)]


def score_metrics(actual_h, actual_a, predictions):
    if not predictions:
        return {}
    top = [(int(h), int(a)) for h, a, _ in predictions]
    ah, aa = int(actual_h), int(actual_a)
    best_h, best_a = top[0]
    return {
        "home_goals_mae": abs(best_h - ah),
        "away_goals_mae": abs(best_a - aa),
        "total_goals_mae": abs((best_h + best_a) - (ah + aa)),
        "exact_score_hit": int((best_h, best_a) == (ah, aa)),
        "top3_score_hit": int((ah, aa) in top[:3]),
        "top4_score_hit": int((ah, aa) in top[:4]),
    }
