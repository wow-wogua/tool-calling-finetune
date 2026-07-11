"""工具调用评测的解析、参数归一化与指标汇总。"""

import json
import re
import statistics


PLATFORM_ALIASES = {
    "b站": "bilibili",
    "哔哩哔哩": "bilibili",
    "bilibili": "bilibili",
    "douyin": "douyin",
    "抖音": "douyin",
    "kuaishou": "kuaishou",
    "快手": "kuaishou",
    "xiaohongshu": "xiaohongshu",
    "小红书": "xiaohongshu",
}


def parse_model_output(text: str) -> tuple[dict, bool]:
    """解析 JSON；允许代码块和前后说明，但单独记录 JSON 有效率。"""
    clean = (text or "").strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        value = json.loads(clean)
        return value if isinstance(value, dict) else {"tool": "parse_error", "params": {}}, isinstance(value, dict)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, dict):
                return value, True
        except json.JSONDecodeError:
            pass
    return {"tool": "parse_error", "params": {}}, False


def _canonical_platform(value):
    if not isinstance(value, str):
        return value
    return PLATFORM_ALIASES.get(value.strip().lower(), value.strip().lower())


def _normalize_text(value: str) -> str:
    return re.sub(r"[\s，。！？、,.!?：:；;\-_/]+", "", value).lower()


def _normalize_keyword(value: str) -> str:
    normalized = _normalize_text(value)
    generic_fragments = (
        "bilibili", "b站", "哔哩哔哩", "douyin", "抖音", "kuaishou", "快手",
        "xiaohongshu", "小红书", "最近", "近期", "当前", "热门", "爆款", "视频",
        "样本", "赛道", "类内容", "内容", "全站", "方向", "相关", "解说", "区",
    )
    for fragment in generic_fragments:
        normalized = normalized.replace(fragment, "")
    return normalized


def _normalize_query(value: str) -> str:
    normalized = _normalize_text(value)
    filler_fragments = (
        "有没有", "有哪些", "可复用", "可以", "帮我", "给我", "先查", "查一下",
        "找一下", "参考", "资料", "内容", "系统", "相关", "内部", "的", "和",
    )
    for fragment in filler_fragments:
        normalized = normalized.replace(fragment, "")
    return normalized


def _equivalent_param(key: str, actual, expected) -> bool:
    if key == "keyword" and isinstance(actual, str) and isinstance(expected, str):
        actual_keyword = _normalize_keyword(actual)
        expected_keyword = _normalize_keyword(expected)
        return actual_keyword == expected_keyword
    if key == "query" and isinstance(actual, str) and isinstance(expected, str):
        actual_query = _normalize_query(actual)
        expected_query = _normalize_query(expected)
        return actual_query == expected_query or actual_query in expected_query or expected_query in actual_query
    return actual == expected


def normalize_params(params: dict | None) -> dict:
    """归一化平台别名、列表顺序、空默认参数和常见字符串格式。"""
    normalized = {}
    for key, value in (params or {}).items():
        if value in (None, "") and key in {"limit", "top_k"}:
            continue
        if key == "platform":
            normalized[key] = _canonical_platform(value)
        elif key == "platforms":
            values = value if isinstance(value, list) else [value]
            normalized[key] = sorted({_canonical_platform(item) for item in values})
        elif isinstance(value, str):
            normalized[key] = _normalize_text(value)
        elif isinstance(value, list):
            normalized[key] = sorted(value, key=lambda item: str(item))
        else:
            normalized[key] = value
    return normalized


def _is_placeholder_identifier(value: str) -> bool:
    normalized = _normalize_text(value)
    return any(fragment in normalized for fragment in (
        "需要", "实际", "具体", "未知", "链接", "videourl", "videoid", "bv号", "请提供",
    ))


def _identifier_hallucinated(actual_tool, actual_params: dict, expected: dict, input_text: str) -> bool:
    required_key = {"get_transcript": "video_url", "get_trend_data": "video_id"}.get(actual_tool)
    if not required_key or required_key in expected:
        return False
    value = actual_params.get(required_key)
    if not isinstance(value, str) or not value or _is_placeholder_identifier(value):
        return False
    return _normalize_text(value) not in _normalize_text(input_text)


def score_tool_call(actual: dict, expected_tool: str | None, expected_params: dict, input_text: str = "") -> dict:
    actual_tool = actual.get("tool")
    canonical_actual = None if actual_tool in (None, "none") else actual_tool
    canonical_expected = expected_tool
    tool_correct = canonical_actual == canonical_expected

    actual_params = normalize_params(actual.get("params", {}))
    expected = normalize_params(expected_params)
    field_results = {
        key: _equivalent_param(key, actual_params.get(key), expected_value)
        for key, expected_value in expected.items()
    }
    params_correct = tool_correct and all(field_results.values())
    identifier_hallucinated = _identifier_hallucinated(
        canonical_actual, actual_params, expected, input_text
    )

    return {
        "tool_correct": tool_correct,
        "params_correct": params_correct,
        "fully_correct": tool_correct and params_correct,
        "safe_call_correct": tool_correct and params_correct and not identifier_hallucinated,
        "identifier_hallucinated": identifier_hallucinated,
        "normalized_actual_params": actual_params,
        "normalized_expected_params": expected,
        "field_results": field_results,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(results: list[dict]) -> dict:
    total = len(results)
    tool_correct = sum(item["tool_correct"] for item in results)
    param_cases = [item for item in results if item.get("normalized_expected_params")]
    params_correct = sum(item["params_correct"] for item in param_cases)
    full_correct = sum(item["fully_correct"] for item in results)
    safe_call_correct = sum(item.get("safe_call_correct", item["fully_correct"]) for item in results)
    identifier_hallucinations = sum(item.get("identifier_hallucinated", False) for item in results)
    json_valid = sum(item["json_valid"] for item in results)
    latencies = [item["latency_ms"] for item in results]
    return {
        "total": total,
        "tool_correct": tool_correct,
        "params_correct": params_correct,
        "param_total": len(param_cases),
        "full_correct": full_correct,
        "safe_call_correct": safe_call_correct,
        "identifier_hallucinations": identifier_hallucinations,
        "json_valid": json_valid,
        "tool_accuracy": tool_correct / total * 100 if total else 0.0,
        "params_accuracy": params_correct / len(param_cases) * 100 if param_cases else 0.0,
        "full_accuracy": full_correct / total * 100 if total else 0.0,
        "safe_call_accuracy": safe_call_correct / total * 100 if total else 0.0,
        "identifier_hallucination_rate": identifier_hallucinations / total * 100 if total else 0.0,
        "json_rate": json_valid / total * 100 if total else 0.0,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
    }
