from __future__ import annotations

import pandas as pd

COMPETITIONS = [
    "EPL", "CHA", "BL1", "SA", "LL", "FL1", "ERE", "UCL", "UEL", "J1", "J2", "J3", "DFBP", "FRI", "CAR"
]

FOOTBALL_DATA_SUPPORTED = {"EPL", "CHA", "BL1", "SA", "LL", "FL1", "ERE"}


def build_coverage(observed: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    observed = observed if observed is not None else pd.DataFrame()
    for c in COMPETITIONS:
        if c not in FOOTBALL_DATA_SUPPORTED:
            rows.append({"competition": c, "source": "Football-Data.co.uk", "status": "UNAVAILABLE", "reason": "No adapter/mapping in current source"})
            continue
        if observed.empty:
            rows.append({"competition": c, "source": "Football-Data.co.uk", "status": "NOT_AUDITED", "reason": "Acquisition not executed"})
        else:
            x = observed[observed.competition == c]
            if x.empty:
                rows.append({"competition": c, "source": "Football-Data.co.uk", "status": "UNAVAILABLE", "reason": "No acquired rows"})
            else:
                rows.append({"competition": c, "source": "Football-Data.co.uk", "status": "AVAILABLE", "rows": int(len(x))})
    return pd.DataFrame(rows)
