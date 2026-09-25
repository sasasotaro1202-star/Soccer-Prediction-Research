from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from src.data.competition_sources import TARGET_COMPETITIONS

TaskName = Literal["1X2", "Score", "O/U", "BTTS", "MOM"]


@dataclass(frozen=True)
class TaskSpec:
    task: TaskName
    output_count: int
    unit: str
    selection_target: str


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec("1X2", 1, "outcome distribution", "chronological OOS / LogLoss+Brier+ECE"),
    TaskSpec("Score", 3, "scoreline candidates", "chronological OOS / score LogLoss+Top3"),
    TaskSpec("O/U", 2, "market probabilities", "chronological OOS / LogLoss+Brier"),
    TaskSpec("BTTS", 2, "market probabilities", "chronological OOS / LogLoss+Brier"),
    TaskSpec("MOM", 4, "player candidates", "chronological OOS / Top4+MRR+calibration"),
)


def build_task_scope_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for competition in TARGET_COMPETITIONS:
        for spec in TASK_SPECS:
            rows.append({
                "competition": competition,
                "task": spec.task,
                "output_count": spec.output_count,
                "unit": spec.unit,
                "selection_target": spec.selection_target,
                "coverage_status": "UNVERIFIED",
                "pit_status": "UNVERIFIED",
                "oos_status": "UNVERIFIED",
                "calibration_status": "UNVERIFIED",
                "robustness_status": "UNVERIFIED",
                "production_eligible": False,
            })
    return rows


def scope_summary() -> dict[str, int]:
    competitions = len(TARGET_COMPETITIONS)
    tasks = len(TASK_SPECS)
    return {
        "competitions": competitions,
        "tasks": tasks,
        "task_cells": competitions * tasks,
        "score_choices_per_match": 3,
        "mom_choices_per_match": 4,
    }
