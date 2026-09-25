from src.research.task_scope import TASK_SPECS, build_task_scope_matrix, scope_summary
from src.data.competition_sources import TARGET_COMPETITIONS


def test_scope_matrix_covers_every_active_competition_and_task():
    matrix = build_task_scope_matrix()
    assert len(matrix) == len(TARGET_COMPETITIONS) * len(TASK_SPECS)
    assert {(r["competition"], r["task"]) for r in matrix} == {
        (c, t.task) for c in TARGET_COMPETITIONS for t in TASK_SPECS
    }


def test_output_cardinality_is_locked():
    specs = {x.task: x.output_count for x in TASK_SPECS}
    assert specs["Score"] == 3
    assert specs["MOM"] == 4
    assert specs["1X2"] == 1
    assert specs["O/U"] == 2
    assert specs["BTTS"] == 2


def test_scope_summary_is_consistent():
    s = scope_summary()
    assert s["competitions"] == 35
    assert s["tasks"] == 5
    assert s["task_cells"] == 175
    assert s["score_choices_per_match"] == 3
    assert s["mom_choices_per_match"] == 4
