from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.football_data import load_available_history
from src.data.pit_source_adapter import apply_pit_evidence, build_pit_diagnostic
from src.data.pit_archive_fallback import apply_arquivo_fallback


def select_sample(history: pd.DataFrame, *, rows_per_season: int = 1, competition: str = "EPL") -> pd.DataFrame:
    """Select a deterministic PIT audit sample from canonical acquired records."""
    if history.empty:
        return history.copy()
    sample = history[history["competition"].astype(str).eq(competition)].copy()
    if sample.empty:
        raise RuntimeError(f"No acquired rows for audit competition: {competition}")
    sample["_kickoff"] = pd.to_datetime(sample["kickoff_utc"], utc=True, errors="coerce")
    sample = sample.sort_values(["season", "_kickoff", "home_team", "away_team"])
    sample = sample.groupby("season", group_keys=False).head(max(1, int(rows_per_season)))
    return sample.drop(columns=["_kickoff"])


def run(out_dir: str = "artifacts/pit-audit", rows_per_season: int = 1, competition: str = "EPL") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = None

    try:
        history, acquisition = load_available_history()
        sample = select_sample(history, rows_per_season=rows_per_season, competition=competition)
        sample.to_csv(out / "pit_audit_sample.csv", index=False)

        diagnostic = build_pit_diagnostic(sample)
        diagnostic.to_csv(out / "pit_audit_diagnostic.csv", index=False)

        # Primary evidence: Internet Archive / Wayback.
        replayed = apply_pit_evidence(sample)
        # Secondary evidence is bounded and fail-closed. It may improve coverage,
        # but it can never turn a transport failure into VERIFIED without an exact
        # archived-result match.
        replayed = apply_arquivo_fallback(replayed)
        replayed.to_csv(out / "pit_audit_replayed.csv", index=False)

        status_counts = replayed["pit_evidence_status"].value_counts(dropna=False).to_dict()
        reason_counts = replayed["pit_evidence_reason"].fillna("").value_counts().to_dict()
        verified = int(replayed["pit_evidence_status"].eq("VERIFIED").sum())
        total = int(len(replayed))
        unverifiable = total - verified
        report = {
            "status": "OK" if total > 0 and verified == total else "BLOCKED",
            "competition": competition,
            "sample_rows": total,
            "verified_rows": verified,
            "unverifiable_rows": unverifiable,
            "verified_rate": float(verified / total) if total else 0.0,
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
            "diagnostic": diagnostic.to_dict(orient="records"),
        }
    except Exception as exc:
        # Never lose the audit result because an archive provider crashed. The
        # result remains BLOCKED and records the exception explicitly.
        try:
            sample_rows = int(len(sample))
        except Exception:
            sample_rows = 0
        report = {
            "status": "BLOCKED",
            "competition": competition,
            "sample_rows": sample_rows,
            "verified_rows": 0,
            "unverifiable_rows": sample_rows,
            "verified_rate": 0.0,
            "status_counts": {"UNVERIFIABLE": sample_rows} if sample_rows else {},
            "reason_counts": {"AUDIT_EXCEPTION": 1},
            "audit_exception_type": type(exc).__name__,
            "audit_exception": str(exc),
        }

    (out / "pit_audit_status.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if report["status"] != "OK":
        raise SystemExit(2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="EPL")
    parser.add_argument("--rows-per-season", type=int, default=1)
    parser.add_argument("--out-dir", default="artifacts/pit-audit")
    args = parser.parse_args()
    run(args.out_dir, args.rows_per_season, args.competition)
