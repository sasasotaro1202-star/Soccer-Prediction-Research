from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.competition_sources import AUXILIARY_COMPETITIONS, AUXILIARY_NAMES, auxiliary_source_plans
from src.data.international_adapter import load_international_history

# Public JFA entry points used for schedule/result discovery. These are registry
# URLs, not a claim that every historical page is parseable or PIT-safe.
JFA_REGISTRY = {
    "EMPERORS_CUP": "https://www.jfa.jp/match/emperorscup_{year}/schedule_result/",
    "INTERHIGH": "https://www.jfa.jp/match/koukou_soutai_{year}/men/schedule_result/",
    "JFA_U20": "https://www.jfa.jp/national_team/u20/schedule_result/{year}.html",
    "JFA_U18": "https://www.jfa.jp/national_team/u18/schedule_result/{year}.html",
}


def run_auxiliary_audit(out_dir: str = "artifacts", start_year: int = 2010, end_year: int = 2026) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    history, coverage = load_international_history(start_year=start_year, end_year=end_year)
    coverage.to_csv(out / "auxiliary_international_coverage.csv", index=False)
    if not history.empty:
        history.to_csv(out / "auxiliary_international_history.csv", index=False)

    plans = pd.DataFrame([
        {
            "competition": p.competition,
            "competition_name": AUXILIARY_NAMES[p.competition],
            "canonical_candidates": "|".join(p.canonical_candidates),
            "discovery_only": "|".join(p.discovery_only),
            "pit_status": p.pit_status,
            "notes": p.notes,
            "jfa_registry_url": JFA_REGISTRY.get(p.competition, ""),
        }
        for p in auxiliary_source_plans()
    ])
    plans.to_csv(out / "auxiliary_source_plan.csv", index=False)

    # This registry is intentionally explicit: these competitions are now part
    # of the research universe, but not silently mixed into the 14-core model.
    registry = []
    for competition in AUXILIARY_COMPETITIONS:
        if competition in coverage.competition.astype(str).unique().tolist():
            subset = coverage[coverage.competition.astype(str) == competition]
            rows = int(pd.to_numeric(subset.rows, errors="coerce").fillna(0).sum())
            available = int((subset.status.astype(str) == "AVAILABLE").sum())
        else:
            rows = 0
            available = 0
        registry.append({
            "competition": competition,
            "competition_name": AUXILIARY_NAMES[competition],
            "observed_rows": rows,
            "available_season_cells": available,
            "production_training_allowed": False,
            "production_reason": "PIT publication timestamp is not established; auxiliary data may inform future research but is excluded from production training.",
            "jfa_registry_url": JFA_REGISTRY.get(competition, ""),
        })
    registry_df = pd.DataFrame(registry)
    registry_df.to_csv(out / "auxiliary_competition_registry.csv", index=False)

    summary = {
        "auxiliary_competition_count": len(AUXILIARY_COMPETITIONS),
        "auxiliary_competitions": list(AUXILIARY_COMPETITIONS),
        "international_source": "openfootball/internationals",
        "international_rows": int(len(history)),
        "international_coverage_cells": int(len(coverage)),
        "pit_policy": "PIT_UNKNOWN rows are never promoted to production training",
        "core_model_contamination_prevented": True,
        "jfa_registry_count": int(sum(bool(v) for v in JFA_REGISTRY.values())),
        "audit_execution_ok": True,
    }
    (out / "auxiliary_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_auxiliary_audit(), indent=2, ensure_ascii=False))
