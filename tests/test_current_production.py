import io
import zipfile
from pathlib import Path

import pytest

from src.prediction.current_production import _safe_extract, manifest_url, release_asset_url


def test_current_production_urls_are_stable():
    repo='sasasotaro1202-star/Soccer-Prediction-Research'
    assert manifest_url(repo).endswith('/main/production/current.json')
    assert release_asset_url(repo).endswith('/production-current/production-current.zip')


@pytest.mark.parametrize('repository', ['', 'invalid', 'a/b/c'])
def test_current_production_rejects_invalid_repository(repository):
    with pytest.raises(ValueError):
        manifest_url(repository)


def test_safe_extract_rejects_zip_slip(tmp_path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as zf:
        zf.writestr('../escape.txt','nope')
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(RuntimeError, match='Unsafe production archive path'):
            _safe_extract(zf, Path(tmp_path))
