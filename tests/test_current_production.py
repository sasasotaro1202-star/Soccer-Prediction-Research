import pytest

from src.prediction.current_production import manifest_url, release_asset_url


def test_current_production_urls_are_stable():
    assert manifest_url('sasasotaro1202-star/Soccer-Prediction-Research').endswith('/main/production/current.json')
    assert release_asset_url('sasasotaro1202-star/Soccer-Prediction-Research').endswith('/production-current/production-current.zip')


@pytest.mark.parametrize('repository', ['', 'invalid', 'a/b/c'])
def test_current_production_rejects_invalid_repository(repository):
    with pytest.raises(ValueError):
        manifest_url(repository)
