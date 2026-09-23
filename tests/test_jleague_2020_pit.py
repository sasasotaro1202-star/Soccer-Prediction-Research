from __future__ import annotations

from src.data import pit_jleague_2020_github as pit


def test_parse_git_log_preserves_immutable_commit_timestamp():
    raw = (
        "a" * 40 + "\t2020-07-01T10:00:00+00:00\n"
        + "b" * 40 + "\t2020-08-01T10:00:00+00:00\n"
        + "b" * 40 + "\t2020-08-01T10:00:00+00:00\n"
        + "malformed\n"
    )
    commits = pit._parse_git_log(raw)
    assert [item["sha"] for item in commits] == ["a" * 40, "b" * 40]
    assert commits[0]["commit"]["committer"]["date"] == "2020-07-01T10:00:00+00:00"


def test_commits_uses_git_transport_when_rest_api_is_rate_limited(monkeypatch, tmp_path):
    expected = [
        {
            "sha": "c" * 40,
            "commit": {"committer": {"date": "2020-09-01T10:00:00+00:00"}},
        }
    ]

    def fail_api(*_args, **_kwargs):
        raise RuntimeError("github_api_rate_limit_exhausted")

    monkeypatch.setattr(pit, "_request_json", fail_api)
    monkeypatch.setattr(pit, "_git_commits", lambda *_args, **_kwargs: expected)

    commits = pit._commits(str(tmp_path), timeout=1)
    assert commits == expected
