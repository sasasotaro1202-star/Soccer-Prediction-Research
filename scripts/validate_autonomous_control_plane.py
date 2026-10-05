"""Validate the GitHub-native autonomous control-plane contract."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/".github"/"automation"/"autonomous_lane_manifest.json"
WORKFLOWS=ROOT/".github"/"workflows"
ORCHESTRATOR=WORKFLOWS/"soccer-autonomous-orchestrator.yml"
RECOVERY=WORKFLOWS/"action-failure-recovery.yml"

def workflow_name(text: str, path: Path) -> str:
    m=re.search(r"(?m)^name:\s*(?:(?:\"([^\"]+)\")|(?:'([^']+)')|([^\n]+))\s*$",text)
    if not m: raise ValueError(f"workflow name missing: {path}")
    return next(g for g in m.groups() if g is not None).strip()

def recovery_names(text: str) -> set[str]:
    m=re.search(r"(?ms)^    workflows:\s*\n(.*?)^    types:\s*\[",text)
    if not m: raise ValueError("action-failure-recovery workflow list is missing")
    out=set()
    for line in m.group(1).splitlines():
        s=line.strip()
        if s.startswith("- "):
            v=s[2:].strip()
            if len(v)>=2 and v[0] in {"'", "\""} and v[-1]==v[0]: v=v[1:-1]
            out.add(v)
    return out

def load_manifest() -> dict[str,Any]:
    payload=json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version")!=1: raise ValueError("unsupported manifest schema_version")
    lanes=payload.get("lanes")
    if not isinstance(lanes,list) or not lanes: raise ValueError("manifest lanes must be non-empty")
    seen=set()
    for lane in lanes:
        if not isinstance(lane,dict): raise ValueError("manifest lane must be object")
        wf=str(lane.get("workflow","")).strip()
        if not wf or wf in seen: raise ValueError(f"duplicate/empty workflow: {wf!r}")
        seen.add(wf)
        cadence=lane.get("cadence_minutes"); gap=lane.get("min_gap_minutes")
        if type(cadence) is not int or cadence<=0 or type(gap) is not int or gap<0 or gap>cadence:
            raise ValueError(f"invalid cadence/min_gap: {wf}")
        path=WORKFLOWS/wf
        if not path.is_file(): raise ValueError(f"autonomous workflow missing: {wf}")
        text=path.read_text(encoding="utf-8")
        workflow_name(text,path)
        if not re.search(r"(?m)^\s*workflow_dispatch\s*:",text): raise ValueError(f"workflow_dispatch missing: {wf}")
        if not re.search(r"(?m)^concurrency:\s*$",text): raise ValueError(f"concurrency missing: {wf}")
    rec=recovery_names(RECOVERY.read_text(encoding="utf-8"))
    missing=[]
    for lane in lanes:
        path=WORKFLOWS[str(lane["workflow"])]
        name=workflow_name(path.read_text(encoding="utf-8"),path)
        if name not in rec: missing.append(name)
    if missing: raise ValueError("failure-recovery coverage missing: "+", ".join(sorted(missing)))
    orch=ORCHESTRATOR.read_text(encoding="utf-8"); recovery=RECOVERY.read_text(encoding="utf-8")
    if ".github/automation/autonomous_lane_manifest.json" not in orch: raise ValueError("orchestrator not pinned to manifest")
    if "|| true" in orch or "|| true" in recovery: raise ValueError("failure-hiding || true forbidden")
    return payload

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--emit-lanes",action="store_true"); a=p.parse_args()
    payload=load_manifest()
    if a.emit_lanes:
        for lane in payload["lanes"]:
            print(f"{lane['workflow']}\t{lane['cadence_minutes']}\t{lane['min_gap_minutes']}")
    else:
        print(f"autonomous_control_plane: PASS lanes={len(payload['lanes'])}")
    return 0
if __name__=="__main__": raise SystemExit(main())
