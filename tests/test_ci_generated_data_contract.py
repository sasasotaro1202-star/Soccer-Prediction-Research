from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_does_not_rebuild_on_generated_artifact_only_pushes():
    text = CI.read_text(encoding="utf-8")
    for path in (
        '      - "artifacts/**"',
        '      - "models/current/**"',
        '      - "data/experience/**"',
        '      - "data/raw/**"',
    ):
        assert path in text
