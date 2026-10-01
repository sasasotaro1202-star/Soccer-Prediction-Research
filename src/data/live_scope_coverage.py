from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.competition_sources import TARGET_COMPETITIONS


def active_target_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize live upcoming-fixture presence for every active target.

    This is snapshot observation coverage, not historical completeness and
    not evidence that an unobserved target has no fixtures outside the fetch horizon.
    """
    coverage: dict[str, dict[str, Any]] = {}
    if frame.empty or "competition" not in frame.columns:
        normalized = pd.Series(dtype="string")
    else:
        normalized = frame["competition"].astype("string").str.strip().str.upper()

    for competition in TARGET_COMPETITIONS:
        if normalized.empty:
            mask = pd.Series(False, index=frame.index)
        else:
            mask = normalized.eq(competition)
        subset = frame.loc[mask]
        source_col = "matchday_source" if "matchday_source" in subset.columns else "source"
        if source_col in subset.columns:
            sources = sorted({
                str(value).strip()
                for value in subset[source_col].dropna().tolist()
                if str(value).strip()
            })
        else:
            sources = []
        coverage[competition] = {
            "upcoming_rows": int(len(subset)),
            "sources": sources,
        }

    observed = [c for c, item in coverage.items() if item["upcoming_rows"] > 0]
    unobserved = [c for c in TARGET_COMPETITIONS if c not in observed]
    count = len(TARGET_COMPETITIONS)
    return {
        "active_target_count": count,
        "targets_with_upcoming_fixtures": len(observed),
        "targets_without_upcoming_fixtures": len(unobserved),
        "upcoming_fixture_presence_pct": round(100.0 * len(observed) / count, 2) if count else 100.0,
        "unobserved_targets": unobserved,
        "per_competition": coverage,
    }
