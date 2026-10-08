from __future__ import annotations

"""Target-isolated OOS artifact extraction.

The score walk-forward engine computes binary O/U and BTTS losses from the
same PIT-safe joint score distribution. This module materializes those metrics
as separate target artifacts so downstream reporting never has to reinterpret
Score columns as O/U or BTTS evidence.
"""

from pathlib import Path

import pandas as pd


def _write_binary_artifact(
    score_oos: pd.DataFrame,
    *,
    probability_prefix: str,
    output_name: str,
    target_name: str,
) -> pd.DataFrame:
    required = {"oos_start", "oos_end", "n", f"{probability_prefix}_logloss", f"{probability_prefix}_brier"}
    missing = sorted(required - set(score_oos.columns))
    if missing:
        raise ValueError(f"{target_name} target extraction missing columns: {missing}")

    out = score_oos[
        ["oos_start", "oos_end", "n", f"{probability_prefix}_logloss", f"{probability_prefix}_brier"]
    ].copy()
    out = out.rename(
        columns={
            f"{probability_prefix}_logloss": "logloss",
            f"{probability_prefix}_brier": "brier",
        }
    )
    out.insert(0, "target", target_name)
    out["evidence_source"] = "chronological_score_distribution_oos"
    out["calibration_status"] = "NOT_EVALUATED"
    out["pit_status"] = "INHERITED_FROM_SCORE_OOS"
    out["oos_status"] = "TARGET_ISOLATED_DERIVATION"
    return out


def write_target_oos_artifacts(score_oos: pd.DataFrame, out_dir: str | Path) -> dict[str, object]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    if score_oos is None or score_oos.empty:
        empty_ou = pd.DataFrame(columns=[
            "target", "oos_start", "oos_end", "n", "logloss", "brier",
            "evidence_source", "calibration_status", "pit_status", "oos_status",
        ])
        empty_btts = empty_ou.copy()
        empty_ou.to_csv(root / "ou_oos_metrics.csv", index=False)
        empty_btts.to_csv(root / "btts_oos_metrics.csv", index=False)
        return {"status": "NO_SCORE_OOS", "ou_blocks": 0, "btts_blocks": 0}

    ou = _write_binary_artifact(
        score_oos,
        probability_prefix="over_2_5",
        output_name="ou_oos_metrics.csv",
        target_name="O/U_2_5",
    )
    btts = _write_binary_artifact(
        score_oos,
        probability_prefix="btts",
        output_name="btts_oos_metrics.csv",
        target_name="BTTS",
    )

    ou.to_csv(root / "ou_oos_metrics.csv", index=False)
    btts.to_csv(root / "btts_oos_metrics.csv", index=False)

    return {
        "status": "PASS",
        "ou_blocks": int(len(ou)),
        "btts_blocks": int(len(btts)),
        "ou_rows": int(pd.to_numeric(ou["n"], errors="coerce").sum()),
        "btts_rows": int(pd.to_numeric(btts["n"], errors="coerce").sum()),
        "derivation": "score_walk_forward_joint_distribution",
        "target_isolation": True,
        "calibration_status": "NOT_EVALUATED",
    }
