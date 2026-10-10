from pathlib import Path


def test_statsmodels_is_research_only():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").lower()
    assert "statsmodels" not in requirements

    for root_name in ("src/prediction", "src/production"):
        root = Path(root_name)
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            content = path.read_text(encoding="utf-8").lower()
            assert "statsmodels" not in content, f"Production path imports external OSS: {path}"


def test_research_adapter_is_outside_production_namespace():
    path = Path("src/research/external_oss/statsmodels_poisson.py")
    assert path.exists()
    assert path.parts[:3] == ("src", "research", "external_oss")
