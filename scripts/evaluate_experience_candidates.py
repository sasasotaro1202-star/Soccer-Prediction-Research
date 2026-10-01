from __future__ import annotations

from src.research.experience_candidate_evaluator import write_evaluation


if __name__ == "__main__":
    result = write_evaluation(
        candidate_path="artifacts/experience_candidate_plan.json",
        cases_path="artifacts/oos_case_diagnostics.csv",
        blocks_path="artifacts/oos_metrics.csv",
        output_path="artifacts/experience_candidate_evaluation.json",
    )
    print(result)
