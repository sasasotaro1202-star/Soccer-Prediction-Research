from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.score_model_selection import (
    select_score_models_by_competition,
    verify_score_models_by_competition,
)


METRICS = ("score_logloss", "over_2_5_logloss", "over_2_5_brier", "btts_logloss", "btts_brier")


def _row(competition: str, block: int, primary: float, dc: float, n: int = 140) -> dict:
    row = {
        "competition": competition,
        "oos_start": f"2025-0{block + 1}-01T00:00:00Z",
        "oos_end": f"2025-0{block + 1}-28T00:00:00Z",
        "n": n,
        "primary_status": "PASS",
        "dc_status": "PASS",
    }
    for metric in METRICS:
        base = primary if metric == "score_logloss" else primary * 0.8
        cand = dc if metric == "score_logloss" else dc * 0.8
        row[metric] = base
        row[f"dixon_coles_{metric}"] = cand
    return row


def test_competition_selector_uses_independent_development_oos():
    rows = [_row("EPL", i, 1.10, 1.00) for i in range(3)]
    rows += [_row("UCL", 0, 1.10, 1.00), _row("UCL", 1, 1.10, 1.00)]
    dev = pd.DataFrame(rows[:5])
    global_selection = {"selected_method": "primary", "status": "KEEP_PRIMARY"}
    selected = select_score_models_by_competition(dev, global_selection, min_blocks=3, min_rows_per_block=120)

    assert selected["competitions"]["EPL"]["selected_method"] == "dixon_coles"
    assert selected["competitions"]["EPL"]["status"] == "ADOPT_CANDIDATE"
    assert selected["competitions"]["UCL"]["selected_method"] == "primary"
    assert selected["competitions"]["UCL"]["status"] == "GLOBAL_FALLBACK"


def test_competition_selector_is_checked_against_untouched_locked_oos():
    dev = pd.DataFrame([_row("EPL", i, 1.10, 1.00) for i in range(3)])
    selection = select_score_models_by_competition(
        dev,
        {"selected_method": "primary", "status": "KEEP_PRIMARY"},
        min_blocks=3,
        min_rows_per_block=120,
    )
    locked = pd.DataFrame([
        _row("EPL", 3, 1.10, 1.01),
        _row("EPL", 4, 1.10, 1.01),
    ])
    verified = verify_score_models_by_competition(
        selection,
        locked,
        max_metric_regression=0.02,
        min_rows_per_block=120,
    )
    assert verified["locked_oos_inspected"] is True
    assert verified["competitions"]["EPL"]["status"] == "PASS"
    assert verified["competitions"]["EPL"]["selected_method"] == "dixon_coles"
