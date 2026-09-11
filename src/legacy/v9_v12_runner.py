"""Concrete loader for the legacy V9 bridge.

The legacy repository is loaded as an external, pinned source tree. This
module does not copy or rewrite backtest.py. The Research Engine supplies the
PIT/OOS-controlled rows; the legacy bridge supplies only its prediction.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from src.legacy.v9_v12_adapter import LegacyEngineAdapter, LegacyPrediction, make_pit_context


def load_v9_adapter(legacy_root: str | os.PathLike[str], random_state: int = 42) -> LegacyEngineAdapter:
    root = Path(legacy_root).resolve()
    source = root / "research_adapter_v9.py"
    if not source.exists():
        raise FileNotFoundError(f"Legacy V9 adapter not found: {source}")
    spec = importlib.util.spec_from_file_location("legacy_research_adapter_v9", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy adapter: {source}")
    module = importlib.util.module_from_spec(spec)
    old_cwd = os.getcwd()
    sys.path.insert(0, str(root))
    try:
        os.chdir(root)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    engine = module.V9ResearchAdapter(random_state=random_state)

    def predict_fn(context: Mapping[str, Any]) -> Mapping[str, Any]:
        row = context.get("legacy_row")
        if row is None:
            raise ValueError("legacy_row is required for concrete V9 execution")
        return engine.predict(row, pit_verified=True, understat=context.get("understat"))

    adapter = LegacyEngineAdapter(predict_fn=predict_fn, source_version="V9")
    setattr(adapter, "legacy_engine", engine)
    return adapter


def predict_v9(adapter: LegacyEngineAdapter, record: Mapping[str, Any]) -> LegacyPrediction:
    """Run V9 through the Research Engine adapter contract."""
    context = make_pit_context(record)
    return adapter.predict(context)


def observe_v9(adapter: LegacyEngineAdapter, record: Mapping[str, Any], **kwargs: Any) -> None:
    """Apply a realized match to the legacy state after evaluation."""
    engine = getattr(adapter, "legacy_engine", None)
    if engine is None:
        raise ValueError("Adapter was not created by load_v9_adapter")
    row = record.get("legacy_row")
    if row is None:
        raise ValueError("legacy_row is required for V9 observation")
    engine.observe_match(row, **kwargs)
