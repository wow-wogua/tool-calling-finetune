"""Run project 2's frozen graph tasks without modifying project 2.

Run each model mode in a fresh Python process so project-2 imports and model
registry state cannot leak between A/B arms. Results are written only to this
repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT2 = Path(r"D:\internship\viral-video-agent")


def gpu_snapshot() -> dict:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).strip().split(",")
        values = [item.strip() for item in output]
        return {
            "utilization_pct": float(values[0]),
            "memory_used_mib": float(values[1]),
            "memory_total_mib": float(values[2]),
            "temperature_c": float(values[3]),
            "power_w": float(values[4]),
        }
    except Exception as exc:
        return {"error": type(exc).__name__}


def trace_agent_duration(trace: dict, name: str) -> float:
    for item in trace.get("agents", []):
        if item.get("agent") == name:
            return float(item.get("duration_s", 0.0))
    return 0.0


async def run(args) -> None:
    project2 = Path(args.project2).resolve()
    if not project2.joinpath("src", "graph", "builder.py").exists():
        raise FileNotFoundError(f"invalid project-2 path: {project2}")
    if args.mode == "v4":
        os.environ["USE_FINETUNED_MODEL"] = "true"
        os.environ["FINETUNED_MODEL_URL"] = args.model_url
    else:
        os.environ["USE_FINETUNED_MODEL"] = "false"
    os.environ["GRAPH_VERSION"] = "v2"
    os.environ["ENABLE_MOCK_TOOLS"] = "false"

    os.chdir(project2)
    sys.path.insert(0, str(project2))
    from src.api.status import result_status
    from src.gateway.model_registry import model_registry
    from src.graph.builder import build_graph
    from src.utils.trace_tracker import trace_tracker

    # Apply the same environment-controlled registry entry used by project 2's
    # src.main, without importing the unrelated FastAPI/auth layer.
    if os.getenv("USE_FINETUNED_MODEL", "").lower() == "true":
        model_registry.register(
            "researcher",
            {
                "provider": "openai",
                "model": "qwen3-tool-calling",
                "base_url": os.getenv("FINETUNED_MODEL_URL", "http://localhost:8002/v1"),
                "api_key": "not-needed",
            },
        )

    all_cases = json.loads(
        project2.joinpath("src", "eval", "cases", "bilibili_mvp_frozen20.json").read_text(encoding="utf-8")
    )
    selected_ids = set(args.case_ids)
    cases = [item for item in all_cases if item["id"] in selected_ids]
    if len(cases) != len(selected_ids):
        found = {item["id"] for item in cases}
        raise ValueError(f"unknown frozen case ids: {sorted(selected_ids - found)}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "researcher_model": "qwen3-tool-calling-v4.1" if args.mode == "v4" else "deepseek-v4-pro",
        "other_agents": "deepseek-v4-pro",
        "project2_path": str(project2),
        "project2_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=project2, text=True
        ).strip(),
        "project2_read_only": True,
        "case_ids": [item["id"] for item in cases],
        "results": [],
    }
    for item in cases:
        trace_tracker.reset()
        graph = build_graph("v2")
        started = time.perf_counter()
        gpu_before = gpu_snapshot()
        try:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "user_id": f"project3-readonly-ab-{args.mode}",
                        "user_request": item["query"],
                        "platforms": ["bilibili"],
                        "workflow_version": "v2",
                        "task_complete": False,
                        "data_sufficient": False,
                        "analysis_confidence": 0.0,
                        "report_final": "",
                    },
                    config={
                        "configurable": {"thread_id": f"project3-{args.mode}-{item['id']}"},
                        "recursion_limit": 50,
                    },
                ),
                timeout=args.timeout,
            )
            elapsed = round(time.perf_counter() - started, 2)
            status, termination_reason = result_status(result)
            trace = trace_tracker.get_summary()
            tool_results = result.get("tool_results", [])
            evidence = result.get("evidence", [])
            row = {
                "case_id": item["id"],
                "query": item["query"],
                "analysis_mode": item["analysis_mode"],
                "status": status,
                "termination_reason": termination_reason,
                "task_completed": status == "completed",
                "elapsed_s": elapsed,
                "researcher_duration_s": trace_agent_duration(trace, "researcher_v2"),
                "llm_calls": trace.get("total_llm_calls", 0),
                "research_tasks": len(result.get("research_tasks", [])),
                "tool_choices": [entry.get("tool") for entry in tool_results],
                "tool_params": [entry.get("params", {}) for entry in tool_results],
                "tool_statuses": [entry.get("status") for entry in tool_results],
                "unavailable_tool_calls": sum(entry.get("status") == "unavailable" for entry in tool_results),
                "invalid_param_calls": sum(entry.get("status") == "invalid_params" for entry in tool_results),
                "evidence_groups": len(evidence),
                "evidence_items": sum(int(entry.get("sample_count", 0)) for entry in evidence),
                "raw_data_count": len(result.get("raw_data", [])),
                "report_length": len(result.get("report_final", "")),
                "trace": trace,
                "gpu_before": gpu_before,
                "gpu_after": gpu_snapshot(),
            }
        except Exception as exc:
            row = {
                "case_id": item["id"],
                "query": item["query"],
                "analysis_mode": item["analysis_mode"],
                "status": "failed",
                "termination_reason": type(exc).__name__,
                "task_completed": False,
                "elapsed_s": round(time.perf_counter() - started, 2),
                "error": str(exc)[:500],
                "trace": trace_tracker.get_summary(),
                "gpu_before": gpu_before,
                "gpu_after": gpu_snapshot(),
            }
        payload["results"].append(row)
        print(json.dumps({
            "mode": args.mode,
            "case_id": row["case_id"],
            "status": row["status"],
            "elapsed_s": row["elapsed_s"],
            "tool_choices": row.get("tool_choices", []),
            "evidence_items": row.get("evidence_items", 0),
        }, ensure_ascii=False), flush=True)
        await asyncio.sleep(1)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT3_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"result={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deepseek", "v4"], required=True)
    parser.add_argument("--project2", default=str(DEFAULT_PROJECT2))
    parser.add_argument("--model-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--case-ids", nargs="+", default=["mvp-01", "mvp-04", "mvp-11"])
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
