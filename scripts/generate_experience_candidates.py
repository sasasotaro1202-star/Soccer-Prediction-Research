from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.experience_candidates import build_experience_candidate_plan

LEDGER = Path("data/experience/prediction_ledger.csv")
OUT = Path("artifacts/experience_candidate_plan.json")
STATUS = Path("artifacts/experience_candidate_status.json")


def main() -> int:
    frame = pd.read_csv(LEDGER) if LEDGER.is_file() and LEDGER.stat().st_size else pd.DataFrame()
    plan = build_experience_candidate_plan(frame)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=2, sort_keys=True, default=str), encoding="utf-8")
    STATUS.write_text(
        json.dumps(
            {
                "status": plan["status"],
                "settled_rows": plan["settled_rows"],
                "candidate_count": len(plan["candidates"]),
                "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
                "research_only": plan["safety_contract"]["research_only"],
                "production_changed": plan["safety_contract"]["production_changed"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
