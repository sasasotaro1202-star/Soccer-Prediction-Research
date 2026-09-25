from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_immutable_action_pin(workflow: str, action: str) -> bool:
    pattern = rf"uses:\s+{re.escape(action)}@[0-9a-f]{{40}}(?:\s|$)"
    return re.search(pattern, workflow) is not None


def test_9h_phase1_exports_replay_and_phase2_consumes_handoff():
    workflow = (REPO_ROOT / '.github/workflows/soccer-9h-autonomous.yml').read_text(encoding='utf-8')
    assert 'cp artifacts/pit_replay_features.csv artifacts/9h/phase1/pit_replay_features.csv' in workflow
    assert _has_immutable_action_pin(workflow, 'actions/download-artifact')
    assert 'PIT_PHASE1_HANDOFF' in workflow
    assert 'gate.get("pit_publication_time_gate") is True' in workflow
    assert 'Rebuild audit artifacts before research when handoff unavailable' in workflow
    assert 'PIT_PHASE1_HANDOFF != \'true\'' in workflow
