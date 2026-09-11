from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.models.baselines import candidates


def run_walk_forward(df: pd.DataFrame, feature_cols: list[str], min_train: int = 300, validation_frac: float = 0.2, oos_block: int = 100, random_state: int = 42):
    d = df.sort_values("kickoff_utc").reset_index(drop=True).copy()
    d = d[d["pit_verified"] == True].reset_index(drop=True)
    if len(d) < min_train + oos_block:
        raise ValueError(f"Not enough PIT-verified rows: {len(d)}; need at least {min_train + oos_block}")
    results, selected = [], []
    start = min_train
    while start < len(d):
        oos_end = min(start + oos_block, len(d))
        train = d.iloc[:start]
        oos = d.iloc[start:oos_end]
        val_n = max(30, int(len(train) * validation_frac))
        fit = train.iloc[:-val_n]
        val = train.iloc[-val_n:]
        Xfit, yfit = fit[feature_cols], fit.target.astype(int)
        Xval, yval = val[feature_cols], val.target.astype(int)
        scores = {}
        for name, model in candidates(random_state).items():
            model.fit(Xfit, yfit)
            scores[name] = classification_metrics(yval, model.predict_proba(Xval))
        best = min(scores, key=lambda k: scores[k]["logloss"])
        selected.append({"oos_start": str(oos.kickoff_utc.min()), "selected_model": best, "validation": scores[best]})
        model = candidates(random_state)[best]
        model.fit(train[feature_cols], train.target.astype(int))
        proba = model.predict_proba(oos[feature_cols])
        m = classification_metrics(oos.target.astype(int), proba)
        results.append({"oos_start": str(oos.kickoff_utc.min()), "oos_end": str(oos.kickoff_utc.max()), "model": best, **m, "n": len(oos)})
        start = oos_end
    return pd.DataFrame(results), pd.DataFrame(selected)
