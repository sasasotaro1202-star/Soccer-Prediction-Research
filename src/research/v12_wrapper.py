"""Compatibility wrapper for legacy V12 autonomous-research artifacts.

V12's research stages remain useful evidence, but this wrapper deliberately
removes promotion authority. The new Research Engine owns PIT, OOS selection,
independent validation, and adoption.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class V12ResearchWrapper:
    """Read legacy V12 research evidence as advisory input only."""

    source_version = "V12"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def available(self) -> bool:
        return (self.root / "research_state.json").exists() or (self.root / "candidate_evaluations.json").exists()

    def _load(self, name: str, default: Any):
        p = self.root / name
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default

    def advisory_evidence(self) -> dict[str, Any]:
        state = self._load("research_state.json", {})
        candidates = self._load("candidate_evaluations.json", [])
        comparison = self._load("model_comparison_v12.json", {})
        return {
            "source_version": self.source_version,
            "available": self.available(),
            "state": state,
            "candidate_evaluations": candidates,
            "comparison": comparison,
            "promotion_authority": False,
        }
