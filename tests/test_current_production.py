from __future__ import annotations

import zipfile

import pytest

from src.prediction.current_production import _safe_extract


def test_safe_extract_rejects_zip_slip(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "no")
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(RuntimeError, match="Unsafe production archive path"):
            _safe_extract(zf, tmp_path / "out")


def test_safe_extract_accepts_nested_files(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("production_model.pkl", b"model")
    root = tmp_path / "out"
    root.mkdir()
    with zipfile.ZipFile(archive) as zf:
        _safe_extract(zf, root)
    assert (root / "production_model.pkl").read_bytes() == b"model"
