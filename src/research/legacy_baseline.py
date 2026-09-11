"""Run legacy V9/V12 baselines under Research Engine OOS control."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from src.legacy.v9_v12_runner import observe_v9, predict_v9


def run_v9_baseline(adapter: Any, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Generate legacy V9 predictions in chronological order.

    The input sequence must already be a locked chronological OOS slice. This
    function never creates a split, never selects a model, and never bypasses
    the adapter's PIT gate. A realized row is observed only after prediction.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        pred = predict_v9(adapter, record)
        probs = pred.probabilities
        row = {
            "match_id": pred.match_id,
            "H": float(probs["H"]),
            "D": float(probs["D"]),
            "A": float(probs["A"]),
            "source_version": pred.source_version,
            "prediction_cutoff_at_utc": pred.prediction_cutoff_at_utc,
            "score_candidates": list(pred.score_candidates),
            "metadata": dict(pred.metadata),
        }
        rows.append(row)
        # Evaluation must occur before this call in the Research Engine. The
        # caller supplies the realized row only as part of the same PIT-safe
        # record, so post-match state cannot influence the current prediction.
        observe_v9(adapter, record)
    return pd.DataFrame(rows)
