from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

POLICY_PATH = Path("config/dynamic_simulator_policy.json")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _exists(root: Path, rel: str) -> bool:
    return (root / rel).is_file()


def inspect(root: Path) -> dict[str, Any]:
    policy = json.loads((root / POLICY_PATH).read_text(encoding="utf-8"))
    prereqs = policy["prerequisites"]
    capabilities = {
        "pit_contract": all(_exists(root, p) for p in prereqs["pit_contract"]),
        "score_distribution": all(_exists(root, p) for p in prereqs["score_distribution"]),
        "oos": all(_exists(root, p) for p in prereqs["oos"]),
        "calibration": _exists(root, prereqs["calibration"]),
        "uncertainty": all(_exists(root, p) for p in prereqs["uncertainty"]),
        "routing": _exists(root, prereqs["routing"]),
        "matchday": _exists(root, prereqs["matchday"]),
        "dynamic_hazard_candidate": _exists(root, prereqs["candidate_dynamic_hazard"]),
    }

    if not capabilities["pit_contract"]:
        stage = "BLOCKED"
        next_action = "repair_PIT_foundation"
    elif not capabilities["score_distribution"] or not capabilities["oos"]:
        stage = "FOUNDATION_GAP"
        next_action = "repair_score_or_OOS_foundation"
    elif not capabilities["dynamic_hazard_candidate"]:
        stage = "DYNAMIC_MATCH_STATE_WAITING"
        next_action = "continue_PIT_verified_dynamic_hazard_research_and_source_discovery"
    else:
        stage = "DYNAMIC_MATCH_STATE_READY"
        next_action = "run_dynamic_hazard_OOS_then_calibration_robustness_before_any_promotion"

    return {
        "schema_version": 1,
        "status": stage,
        "next_action": next_action,
        "main_sha": _git_sha(),
        "capabilities": capabilities,
        "research_only": True,
        "production_changed": False,
        "production_usable": False,
        "promotion_candidate": False,
        "performance_verified": False,
        "frozen_holdout_access_allowed": False,
        "pit_required": True,
        "pit_status": "REQUIRED_FOR_ANY_PERFORMANCE_CLAIM",
        "policy": "config/dynamic_simulator_policy.json",
        "reference": "docs/HIERARCHICAL_DYNAMIC_SIMULATOR.md",
    }


def write_status(root: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    status = inspect(root)
    (out / "dynamic_simulator_frontier_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="artifacts/dynamic_simulator_research")
    args = parser.parse_args()
    status = write_status(Path(args.root), Path(args.out))
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
