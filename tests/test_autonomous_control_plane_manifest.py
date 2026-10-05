from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def test_autonomous_lane_manifest_is_valid():
    result=subprocess.run([sys.executable,"scripts/validate_autonomous_control_plane.py","--emit-lanes"],cwd=ROOT,check=True,capture_output=True,text=True)
    rows=[x.split("|") for x in result.stdout.splitlines() if x.strip()]
    payload=json.loads((ROOT/".github"/"automation"/"autonomous_lane_manifest.json").read_text(encoding="utf-8"))
    assert len(rows)==len(payload["lanes"])
    assert {r[0] for r in rows}=={lane["workflow"] for lane in payload["lanes"]}
def test_autonomous_lane_files_exist():
    assert (ROOT/".github"/"automation"/"autonomous_lane_manifest.json").is_file()
    assert (ROOT/"scripts"/"validate_autonomous_control_plane.py").is_file()
def test_control_plane_has_no_failure_hiding():
    for name in ("soccer-autonomous-orchestrator.yml","action-failure-recovery.yml"):
        assert "|| true" not in (ROOT/".github"/"workflows"/name).read_text(encoding="utf-8")
