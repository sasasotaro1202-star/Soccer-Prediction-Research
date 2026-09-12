from src.data.pit_archive_fallback import _normalise_cdx_payload


def test_normalise_cdx_list_of_lists():
    payload = [["timestamp", "original"], ["20200101000000", "https://example.test/a"]]
    assert _normalise_cdx_payload(payload) == [{"timestamp": "20200101000000", "original": "https://example.test/a"}]


def test_normalise_cdx_list_of_dicts():
    payload = [{"timestamp": "20200101000000", "original": "https://example.test/a"}]
    assert _normalise_cdx_payload(payload) == payload


def test_normalise_cdx_mapping_data():
    payload = {"data": [{"timestamp": "20200101000000", "original": "https://example.test/a"}]}
    assert _normalise_cdx_payload(payload) == payload["data"]


def test_normalise_cdx_mapping_rows_as_matrix():
    payload = {"rows": [["timestamp", "original"], ["20200101000000", "https://example.test/a"]]}
    assert _normalise_cdx_payload(payload) == [{"timestamp": "20200101000000", "original": "https://example.test/a"}]


def test_normalise_cdx_invalid_payload_is_empty():
    assert _normalise_cdx_payload({"unexpected": {}}) == []
    assert _normalise_cdx_payload(None) == []
