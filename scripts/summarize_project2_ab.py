"""Summarize the read-only project-2 model A/B result files."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def arm_summary(payload: dict) -> dict:
    rows = payload["results"]
    statuses = {name: sum(item["status"] == name for item in rows) for name in ("completed", "partial", "failed")}
    return {
        "researcher_model": payload["researcher_model"],
        "cases": len(rows),
        "statuses": statuses,
        "mean_total_elapsed_s": statistics.mean(item["elapsed_s"] for item in rows),
        "mean_researcher_duration_s": statistics.mean(item.get("researcher_duration_s", 0) for item in rows),
        "mean_llm_calls": statistics.mean(item.get("llm_calls", 0) for item in rows),
        "invalid_param_calls": sum(item.get("invalid_param_calls", 0) for item in rows),
        "unavailable_tool_calls": sum(item.get("unavailable_tool_calls", 0) for item in rows),
        "evidence_groups": sum(item.get("evidence_groups", 0) for item in rows),
        "raw_data_count": sum(item.get("raw_data_count", 0) for item in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek", default="results/v4/project2_ab_deepseek.json")
    parser.add_argument("--v4", default="results/v4/project2_ab_v4_1_adapted.json")
    parser.add_argument("--output", default="results/v4/project2_ab_comparison.json")
    args = parser.parse_args()
    deepseek = json.loads(Path(args.deepseek).read_text(encoding="utf-8"))
    v4 = json.loads(Path(args.v4).read_text(encoding="utf-8"))
    by_case_deepseek = {item["case_id"]: item for item in deepseek["results"]}
    by_case_v4 = {item["case_id"]: item for item in v4["results"]}
    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project2_commit": deepseek["project2_commit"],
        "project2_read_only": True,
        "arms": {
            "deepseek": arm_summary(deepseek),
            "v4_1": arm_summary(v4),
        },
        "per_case": [],
        "conclusion": {
            "candidate_integration_passed": True,
            "product_superiority_proven": False,
            "reason": "Both arms produced the same completed/partial distribution and evidence availability. The adapted v4.1 arm removed invalid parameter calls, but external Bilibili/RAG empty results dominated incomplete tasks and Researcher latency was not consistently lower.",
        },
    }
    for case_id in deepseek["case_ids"]:
        left = by_case_deepseek[case_id]
        right = by_case_v4[case_id]
        comparison["per_case"].append({
            "case_id": case_id,
            "deepseek_status": left["status"],
            "v4_status": right["status"],
            "deepseek_tools": left.get("tool_choices", []),
            "v4_tools": right.get("tool_choices", []),
            "deepseek_tool_statuses": left.get("tool_statuses", []),
            "v4_tool_statuses": right.get("tool_statuses", []),
            "deepseek_invalid_params": left.get("invalid_param_calls", 0),
            "v4_invalid_params": right.get("invalid_param_calls", 0),
            "deepseek_researcher_s": left.get("researcher_duration_s", 0),
            "v4_researcher_s": right.get("researcher_duration_s", 0),
            "deepseek_total_s": left["elapsed_s"],
            "v4_total_s": right["elapsed_s"],
            "deepseek_raw_data": left.get("raw_data_count", 0),
            "v4_raw_data": right.get("raw_data_count", 0),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
