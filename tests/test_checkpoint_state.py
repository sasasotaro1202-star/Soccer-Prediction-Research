from src.research.checkpoint_state import (
    SUCCESS, FAILURE, append_checkpoint, can_reuse_success, config_hash, make_checkpoint
)


def test_checkpoint_is_idempotent_for_exact_success(tmp_path):
    path = tmp_path / "checkpoints.jsonl"
    cfg = {"phase": 1, "sources": ["statsbomb", "skillcorner"]}
    cp = make_checkpoint(
        experiment_key="source_probe",
        status=SUCCESS,
        git_sha="abc123",
        run_id=10,
        configuration=cfg,
        artifacts=["a.json"],
    )
    assert append_checkpoint(path, cp) == "INSERTED"
    assert append_checkpoint(path, cp) == "EXISTING_SUCCESS"
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert can_reuse_success(path, experiment_key="source_probe", git_sha="abc123", configuration=cfg)


def test_failed_checkpoint_never_becomes_reusable(tmp_path):
    path = tmp_path / "checkpoints.jsonl"
    cfg = {"phase": 2}
    cp = make_checkpoint(
        experiment_key="pit",
        status=FAILURE,
        git_sha="abc123",
        run_id=11,
        configuration=cfg,
        reason="pit gate blocked",
    )
    assert append_checkpoint(path, cp) == "INSERTED"
    assert not can_reuse_success(path, experiment_key="pit", git_sha="abc123", configuration=cfg)
    assert config_hash(cfg) != config_hash({"phase": 3})


def test_success_does_not_cross_commit_boundary(tmp_path):
    path = tmp_path / "checkpoints.jsonl"
    cfg = {"x": 1}
    cp = make_checkpoint(
        experiment_key="oos",
        status=SUCCESS,
        git_sha="old",
        run_id=1,
        configuration=cfg,
    )
    append_checkpoint(path, cp)
    assert not can_reuse_success(path, experiment_key="oos", git_sha="new", configuration=cfg)
