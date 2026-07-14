"""Smoke-test the local OpenAI-compatible v4 service without exposing secrets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from researcher_prompt_v4 import DEFAULT_CAPABILITIES, build_researcher_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", default="results/v4/openai_service_smoke.json")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    cases = [
        {
            "id": "full_project2_prompt",
            "message": build_researcher_prompt(
                "从B站找近期机器人视频样本",
                ["bilibili"],
                DEFAULT_CAPABILITIES,
                "contract",
            ),
            "expected_tool": "search_videos",
        },
        {
            "id": "bare_task_wrapped_once",
            "message": "不用搜索，直接解释什么是完播率",
            "expected_tool": "none",
        },
        {
            "id": "rag_top_k_schema_clamp",
            "message": build_researcher_prompt(
                "从知识库检索AI编程标题方法论，返回20条参考",
                ["bilibili"],
                DEFAULT_CAPABILITIES,
                "contract",
            ),
            "expected_tool": "rag_search",
            "max_top_k": 10,
        },
    ]
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "health": None,
        "models": None,
        "cases": [],
        "passed": False,
    }
    with httpx.Client(timeout=120) as client:
        payload["health"] = client.get(f"{base_url}/health").json()
        payload["models"] = client.get(f"{base_url}/v1/models").json()
        for item in cases:
            response = client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "qwen3-tool-calling",
                    "messages": [{"role": "user", "content": item["message"]}],
                    "temperature": 0,
                    "max_tokens": 96,
                },
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            passed = parsed.get("tool") == item["expected_tool"]
            if item.get("max_top_k") is not None:
                passed = passed and parsed.get("params", {}).get("top_k", 0) <= item["max_top_k"]
            payload["cases"].append({
                "id": item["id"],
                "expected_tool": item["expected_tool"],
                "actual": parsed,
                "passed": passed,
                "usage": body.get("usage", {}),
            })
    payload["passed"] = (
        payload["health"].get("status") == "ok"
        and all(item["passed"] for item in payload["cases"])
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
