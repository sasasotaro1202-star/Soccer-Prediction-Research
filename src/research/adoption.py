from __future__ import annotations

import pandas as pd


def adoption_decision(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Deterministic gate. Lower LogLoss/Brier/ECE is better; higher accuracy is better."""
    if baseline.empty or candidate.empty:
        return {"status": "HOLD", "reason": "Missing OOS comparison"}
    b = baseline.mean(numeric_only=True); c = candidate.mean(numeric_only=True)
    primary = c["logloss"] < b["logloss"]
    secondary = c["brier"] <= b["brier"] and c["ece"] <= b["ece"] and c["accuracy"] >= b["accuracy"]
    status = "ADOPT" if primary and secondary else "REJECT"
    return {"status": status, "baseline": b.to_dict(), "candidate": c.to_dict(), "primary_logloss_improved": bool(primary), "secondary_ok": bool(secondary)}
