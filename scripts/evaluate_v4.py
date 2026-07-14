"""Deterministic v4 evaluation for base and direct LoRA adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from researcher_prompt_v4 import build_researcher_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = Path.home() / ".cache" / "modelscope" / "Qwen" / "Qwen3-4B"
SPLIT_FILES = {
    "dev": PROJECT_ROOT / "data" / "eval" / "v4_dev.json",
    "hard": PROJECT_ROOT / "data" / "eval" / "v4_hard.json",
    "holdout": PROJECT_ROOT / "data" / "eval" / "v4_holdout.json",
    "capability_holdout": PROJECT_ROOT / "data" / "eval" / "v4_capability_holdout.json",
}
PLATFORM_ALIASES = {
    "b站": "bilibili", "哔哩哔哩": "bilibili", "bilibili": "bilibili",
    "抖音": "douyin", "douyin": "douyin", "快手": "kuaishou", "kuaishou": "kuaishou",
    "小红书": "xiaohongshu", "xiaohongshu": "xiaohongshu",
}


def load_model(base_model: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="left", local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization,
        local_files_only=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    return tokenizer, model


def strict_json(text: str) -> tuple[dict, bool]:
    clean = (text or "").strip()
    try:
        value = json.loads(clean)
        if isinstance(value, dict):
            return value, True
    except json.JSONDecodeError:
        pass
    flexible = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if flexible.startswith("```"):
        flexible = flexible.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    match = re.search(r"\{.*\}", flexible, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, dict):
                return value, False
        except json.JSONDecodeError:
            pass
    return {"tool": "parse_error", "params": {}}, False


def normalize_text(value: str) -> str:
    return re.sub(r"[\s，。！？、,.!?：:；;\-_/]+", "", (value or "").lower())


def normalize_platform(value):
    if not isinstance(value, str):
        return value
    return PLATFORM_ALIASES.get(value.strip().lower(), value.strip().lower())


def normalize_params(params: dict | None) -> dict:
    normalized = {}
    for key, value in (params or {}).items():
        if key == "platform":
            normalized[key] = normalize_platform(value)
        elif key == "platforms":
            values = value if isinstance(value, list) else [value]
            normalized[key] = sorted({normalize_platform(item) for item in values})
        elif isinstance(value, str):
            normalized[key] = value.strip()
        else:
            normalized[key] = value
    return normalized


def semantic_string(key: str, value: str) -> str:
    normalized = normalize_text(value)
    if key == "keyword":
        for fragment in (
            "bilibili", "b站", "哔哩哔哩", "最近", "近期", "当前", "热门", "爆款",
            "视频", "样本", "内容", "赛道", "分区", "频道", "区",
        ):
            normalized = normalized.replace(fragment, "")
    elif key == "query":
        for fragment in (
            "查一下", "检索", "知识库", "资料库", "内部", "参考", "资料", "内容",
            "有没有", "给我", "找一份", "关于", "相关", "的", "和", "对比",
        ):
            normalized = normalized.replace(fragment, "")
        normalized = normalized.replace("过去", "历史")
    return normalized


def value_matches(key: str, actual, expected) -> bool:
    if key in {"keyword", "query"} and isinstance(actual, str) and isinstance(expected, str):
        left = semantic_string(key, actual)
        right = semantic_string(key, expected)
        return left == right or (left and right and (left in right or right in left))
    if key == "platforms":
        return sorted(actual or []) == sorted(expected or [])
    return actual == expected


def score_case(case: dict, actual: dict, json_valid: bool) -> dict:
    expected_tool = case["expected_tool"]
    actual_tool = actual.get("tool")
    if actual_tool in (None, "none"):
        actual_tool = "none"
    runtime_params = dict(actual.get("params") if isinstance(actual.get("params"), dict) else {})
    # Mirror project 2's Pydantic defaults and v2 rag platform injection before scoring.
    if actual_tool == "search_videos":
        runtime_params.setdefault("keyword", "")
        runtime_params.setdefault("platforms", ["bilibili"])
        runtime_params.setdefault("limit", 10)
    elif actual_tool == "rag_search":
        runtime_params.setdefault("top_k", 5)
        if "platform" not in runtime_params and len(case.get("platforms", [])) == 1:
            runtime_params["platform"] = case["platforms"][0]
    actual_params = normalize_params(runtime_params)
    expected_params = normalize_params(case["expected_params"])
    tool_correct = actual_tool == expected_tool
    field_results = {
        key: value_matches(key, actual_params.get(key), expected_value)
        for key, expected_value in expected_params.items()
    }
    params_correct = tool_correct and all(field_results.values())
    if expected_tool == "none":
        params_correct = tool_correct and not actual_params

    tool_state = case["capabilities"].get(actual_tool, {})
    unavailable_violation = actual_tool not in {"none", "parse_error"} and not tool_state.get("enabled", False)
    target_platforms = {normalize_platform(item) for item in case.get("platforms", [])}
    actual_platforms = set(actual_params.get("platforms", []))
    unsupported_platform_violation = False
    if actual_tool == "search_videos":
        unsupported_platform_violation = (
            bool(target_platforms - {"bilibili"})
            or bool(actual_platforms - {"bilibili"})
        )
    if actual_tool == "get_transcript":
        url = actual_params.get("video_url", "")
        if isinstance(url, str) and url:
            unsupported_platform_violation = not (
                url.startswith("https://www.bilibili.com/") or url.startswith("https://b23.tv/")
            )

    hallucinated_identifier = False
    for key in ("video_url", "video_id"):
        value = actual_params.get(key)
        if isinstance(value, str) and value and value not in case["input"]:
            hallucinated_identifier = True

    fully_correct = tool_correct and params_correct
    safe_correct = fully_correct and not (
        unavailable_violation or unsupported_platform_violation or hallucinated_identifier
    )
    capability_relevant = bool(set(case.get("safety_tags", [])) & {
        "capability_focus", "unavailable_tool", "unsupported_platform", "missing_url", "missing_id"
    })
    capability_correct = (
        tool_correct and not unavailable_violation and not unsupported_platform_violation
        and not hallucinated_identifier
    ) if capability_relevant else None
    error_types = []
    if not json_valid:
        error_types.append("invalid_json")
    if not tool_correct:
        error_types.append("wrong_tool")
    elif not params_correct:
        error_types.append("wrong_params")
    if unavailable_violation:
        error_types.append("unavailable_tool_violation")
    if unsupported_platform_violation:
        error_types.append("unsupported_platform_violation")
    if hallucinated_identifier:
        error_types.append("hallucinated_id_or_url")
    return {
        "actual_tool": actual_tool,
        "actual_params": actual_params,
        "normalized_expected_params": expected_params,
        "field_results": field_results,
        "tool_correct": tool_correct,
        "params_correct": params_correct,
        "fully_correct": fully_correct,
        "safe_correct": safe_correct,
        "json_valid": json_valid,
        "unavailable_tool_violation": unavailable_violation,
        "unsupported_platform_violation": unsupported_platform_violation,
        "hallucinated_id_or_url": hallucinated_identifier,
        "capability_relevant": capability_relevant,
        "capability_correct": capability_correct,
        "error_types": error_types,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = round((len(values) - 1) * fraction)
    return values[index]


def summarize(details: list[dict]) -> dict:
    total = len(details)
    param_cases = [item for item in details if item["expected_tool"] != "none"]
    capability_cases = [item for item in details if item["capability_relevant"]]
    latencies = [item["latency_ms"] for item in details]

    def rate(field: str, rows: list[dict] | None = None) -> float:
        selected = details if rows is None else rows
        return sum(bool(item[field]) for item in selected) / len(selected) * 100 if selected else 0.0

    per_tool = {}
    for tool in sorted({item["expected_tool"] for item in details}):
        rows = [item for item in details if item["expected_tool"] == tool]
        per_tool[tool] = {
            "total": len(rows),
            "tool_accuracy": rate("tool_correct", rows),
            "full_accuracy": rate("fully_correct", rows),
            "safe_accuracy": rate("safe_correct", rows),
        }
    per_profile = {}
    for profile in sorted({item["capability_profile"] for item in details}):
        rows = [item for item in details if item["capability_profile"] == profile]
        per_profile[profile] = {
            "total": len(rows),
            "tool_accuracy": rate("tool_correct", rows),
            "full_accuracy": rate("fully_correct", rows),
            "safe_accuracy": rate("safe_correct", rows),
        }
    return {
        "total": total,
        "tool_accuracy": rate("tool_correct"),
        "params_accuracy": rate("params_correct", param_cases),
        "full_accuracy": rate("fully_correct"),
        "safe_accuracy": rate("safe_correct"),
        "json_valid_rate": rate("json_valid"),
        "unavailable_tool_violation_rate": rate("unavailable_tool_violation"),
        "unsupported_platform_violation_rate": rate("unsupported_platform_violation"),
        "hallucinated_id_or_url_rate": rate("hallucinated_id_or_url"),
        "capability_state_accuracy": rate("capability_correct", capability_cases),
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "error_types": dict(sorted(Counter(
            error for item in details for error in item["error_types"]
        ).items())),
        "per_tool": per_tool,
        "per_capability_profile": per_profile,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def api_usage_summary(
    details: list[dict], input_cost_per_million: float, output_cost_per_million: float
) -> dict:
    input_tokens = sum(item.get("usage", {}).get("input_tokens", 0) for item in details)
    output_tokens = sum(item.get("usage", {}).get("output_tokens", 0) for item in details)
    retries = sum(item.get("retries", 0) for item in details)
    attempts = sum(item.get("attempts", 1) for item in details)
    estimated_cost = (
        input_tokens * input_cost_per_million / 1_000_000
        + output_tokens * output_cost_per_million / 1_000_000
    )
    return {
        "successful_requests": len(details),
        "attempts": attempts,
        "retries": retries,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "pricing_usd_per_million_tokens": {
            "input": input_cost_per_million,
            "output": output_cost_per_million,
        },
    }


def load_api_checkpoint(
    path: Path,
    model: str,
    prompt_variant: str,
    max_tokens: int,
    split_hashes: dict[str, str],
) -> dict:
    expected = {
        "schema_version": 1,
        "model": model,
        "prompt_variant": prompt_variant,
        "max_tokens": max_tokens,
        "split_hashes": split_hashes,
    }
    if not path.exists():
        return {
            **expected,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed": {},
            "failed_attempts": [],
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"checkpoint mismatch for {key}: expected {value!r}, got {checkpoint.get(key)!r}"
            )
    checkpoint.setdefault("completed", {})
    checkpoint.setdefault("failed_attempts", [])
    return checkpoint


def api_generate(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    case_key: str,
    checkpoint: dict,
    checkpoint_path: Path,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[str, float, dict, int]:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )

    for attempt in range(1, max_retries + 2):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            content = response.choices[0].message.content or ""
            usage = response.usage
            usage_payload = {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            return str(content), latency_ms, usage_payload, attempt
        except AuthenticationError:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "case_key": case_key,
                "attempt": attempt,
                "error_type": "authentication_error",
                "retryable": False,
            }
            checkpoint["failed_attempts"].append(event)
            checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(checkpoint_path, checkpoint)
            raise RuntimeError("DeepSeek authentication failed; check the configured API key")
        except (RateLimitError, APITimeoutError, APIConnectionError) as error:
            retryable = attempt <= max_retries
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "case_key": case_key,
                "attempt": attempt,
                "error_type": type(error).__name__,
                "retryable": retryable,
            }
            checkpoint["failed_attempts"].append(event)
            checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(checkpoint_path, checkpoint)
            if not retryable:
                raise RuntimeError(f"DeepSeek request failed after {attempt} attempts: {type(error).__name__}")
        except APIStatusError as error:
            status_code = int(getattr(error, "status_code", 0) or 0)
            retryable = status_code >= 500 and attempt <= max_retries
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "case_key": case_key,
                "attempt": attempt,
                "error_type": "api_status_error",
                "status_code": status_code,
                "retryable": retryable,
            }
            checkpoint["failed_attempts"].append(event)
            checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(checkpoint_path, checkpoint)
            if not retryable:
                raise RuntimeError(f"DeepSeek API returned non-retryable status {status_code}")
        delay = retry_backoff_seconds * (2 ** (attempt - 1))
        print(
            f"[{case_key}] retrying after attempt {attempt}; wait={delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def evaluate_api_splits(client, args, selected: list[str]) -> dict:
    checkpoint_path = Path(args.checkpoint)
    split_hashes = {split: sha256_file(SPLIT_FILES[split]) for split in selected}
    checkpoint = load_api_checkpoint(
        checkpoint_path,
        args.api_model,
        args.prompt_variant,
        args.max_new_tokens,
        split_hashes,
    )
    split_payloads = {}
    all_details = []
    for split in selected:
        cases = json.loads(SPLIT_FILES[split].read_text(encoding="utf-8"))
        if args.limit:
            cases = cases[:args.limit]
        details = []
        for index, item in enumerate(cases, start=1):
            case_key = f"{split}:{item['id']}"
            if case_key in checkpoint["completed"]:
                detail = checkpoint["completed"][case_key]
                print(
                    f"[{split} {index:02d}/{len(cases)}] RESUME "
                    f"expected={item['expected_tool']} actual={detail['actual_tool']}",
                    flush=True,
                )
            else:
                prompt = build_researcher_prompt(
                    item["input"], item["platforms"], item["capabilities"], args.prompt_variant
                )
                response, latency_ms, usage, attempts = api_generate(
                    client,
                    args.api_model,
                    prompt,
                    args.max_new_tokens,
                    case_key,
                    checkpoint,
                    checkpoint_path,
                    args.max_retries,
                    args.retry_backoff_seconds,
                )
                actual, json_valid = strict_json(response)
                scored = score_case(item, actual, json_valid)
                detail = {
                    "repeat": 1,
                    "case": index,
                    "id": item["id"],
                    "input": item["input"],
                    "platforms": item["platforms"],
                    "capability_profile": item["capability_profile"],
                    "intent_family": item["intent_family"],
                    "expected_tool": item["expected_tool"],
                    "expected_params": item["expected_params"],
                    "raw_output": response,
                    "latency_ms": round(latency_ms, 2),
                    "usage": usage,
                    "attempts": attempts,
                    "retries": attempts - 1,
                    **scored,
                }
                checkpoint["completed"][case_key] = detail
                checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(checkpoint_path, checkpoint)
                status = "PASS" if detail["fully_correct"] else "FAIL"
                print(
                    f"[{split} {index:02d}/{len(cases)}] {status} "
                    f"expected={item['expected_tool']} actual={detail['actual_tool']} "
                    f"{latency_ms:.0f}ms tokens={usage['input_tokens']}+{usage['output_tokens']}",
                    flush=True,
                )
            details.append(detail)
        split_payloads[split] = {
            "summary": summarize(details),
            "usage": api_usage_summary(
                details, args.input_cost_per_million, args.output_cost_per_million
            ),
            "details": details,
        }
        all_details.extend(details)
    total_usage = api_usage_summary(
        all_details, args.input_cost_per_million, args.output_cost_per_million
    )
    total_usage["failed_attempts"] = len(checkpoint["failed_attempts"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "provider": "deepseek_openai_compatible_api",
        "model": args.api_model,
        "prompt_variant": args.prompt_variant,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_tokens": args.max_new_tokens,
            "thinking": {"type": "disabled"},
            "repeats": 1,
        },
        "checkpoint": str(checkpoint_path),
        "split_hashes": split_hashes,
        "usage": total_usage,
        "splits": split_payloads,
    }


def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    import torch

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), latency_ms


def evaluate_split(tokenizer, model, split: str, cases: list[dict], prompt_variant: str, repeats: int, max_new_tokens: int) -> dict:
    details = []
    for repeat in range(1, repeats + 1):
        for index, item in enumerate(cases, start=1):
            prompt = build_researcher_prompt(
                item["input"], item["platforms"], item["capabilities"], prompt_variant
            )
            response, latency_ms = generate(tokenizer, model, prompt, max_new_tokens)
            actual, json_valid = strict_json(response)
            scored = score_case(item, actual, json_valid)
            detail = {
                "repeat": repeat,
                "case": index,
                "id": item["id"],
                "input": item["input"],
                "platforms": item["platforms"],
                "capability_profile": item["capability_profile"],
                "intent_family": item["intent_family"],
                "expected_tool": item["expected_tool"],
                "expected_params": item["expected_params"],
                "raw_output": response,
                "latency_ms": round(latency_ms, 2),
                **scored,
            }
            details.append(detail)
            status = "PASS" if detail["fully_correct"] else "FAIL"
            print(
                f"[{split} r{repeat} {index:02d}/{len(cases)}] {status} "
                f"expected={item['expected_tool']} actual={detail['actual_tool']} {latency_ms:.0f}ms",
                flush=True,
            )
    return {"summary": summarize(details), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter")
    parser.add_argument("--label")
    parser.add_argument("--prompt-variant", choices=["contract", "rules"], default="contract")
    parser.add_argument("--splits", default="dev,hard,holdout,capability_holdout")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rescore-input")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--api-model", default="deepseek-v4-pro")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-base-url-env", default="DEEPSEEK_BASE_URL")
    parser.add_argument("--checkpoint", default="results/v4/deepseek_v4_pro_checkpoint.json")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--input-cost-per-million", type=float, default=0.435)
    parser.add_argument("--output-cost-per-million", type=float, default=0.87)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.rescore_input:
        source = json.loads(Path(args.rescore_input).read_text(encoding="utf-8"))
        rescored = dict(source)
        rescored["timestamp"] = datetime.now(timezone.utc).isoformat()
        rescored["label"] = args.label or f"{source.get('label', 'result')}_rescored"
        rescored["rescored_from"] = str(Path(args.rescore_input))
        for split, split_payload in source["splits"].items():
            cases = {item["id"]: item for item in json.loads(SPLIT_FILES[split].read_text(encoding="utf-8"))}
            details = []
            for old in split_payload["details"]:
                item = cases[old["id"]]
                actual, json_valid = strict_json(old["raw_output"])
                scored = score_case(item, actual, json_valid)
                details.append({
                    **old,
                    "actual_tool": scored["actual_tool"],
                    "actual_params": scored["actual_params"],
                    "normalized_expected_params": scored["normalized_expected_params"],
                    "field_results": scored["field_results"],
                    "tool_correct": scored["tool_correct"],
                    "params_correct": scored["params_correct"],
                    "fully_correct": scored["fully_correct"],
                    "safe_correct": scored["safe_correct"],
                    "json_valid": scored["json_valid"],
                    "unavailable_tool_violation": scored["unavailable_tool_violation"],
                    "unsupported_platform_violation": scored["unsupported_platform_violation"],
                    "hallucinated_id_or_url": scored["hallucinated_id_or_url"],
                    "capability_relevant": scored["capability_relevant"],
                    "capability_correct": scored["capability_correct"],
                    "error_types": scored["error_types"],
                })
            rescored["splits"][split] = {"summary": summarize(details), "details": details}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rescored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "label": rescored["label"],
            "output": str(output),
            "summaries": {name: value["summary"] for name, value in rescored["splits"].items()},
        }, ensure_ascii=False, indent=2))
        return
    if not args.label:
        parser.error("--label is required unless --rescore-input is used")
    selected = [item.strip() for item in args.splits.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(SPLIT_FILES))
    if unknown:
        parser.error(f"unknown splits: {unknown}")
    if args.api:
        if args.repeats != 1:
            parser.error("API evaluation supports exactly one repeat")
        if args.adapter:
            parser.error("--adapter cannot be combined with --api")
        api_key = os.getenv(args.api_key_env, "").strip()
        base_url = os.getenv(args.api_base_url_env, "").strip()
        if not api_key:
            parser.error(f"missing API key in environment variable {args.api_key_env}")
        if not base_url:
            parser.error(f"missing API base URL in environment variable {args.api_base_url_env}")
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.api_timeout_seconds)
        payload = evaluate_api_splits(client, args, selected)
        output = Path(args.output)
        write_json_atomic(output, payload)
        print(json.dumps({
            "label": payload["label"],
            "output": str(output),
            "usage": payload["usage"],
            "summaries": {name: value["summary"] for name, value in payload["splits"].items()},
        }, ensure_ascii=False, indent=2))
        return
    if not Path(args.base_model).joinpath("config.json").exists():
        parser.error(f"base model is incomplete: {args.base_model}")
    if args.adapter and not Path(args.adapter).joinpath("adapter_config.json").exists():
        parser.error(f"adapter is incomplete: {args.adapter}")

    tokenizer, model = load_model(args.base_model, args.adapter)
    # One unscored warmup keeps latency comparisons fair.
    generate(tokenizer, model, build_researcher_prompt("直接解释什么是完播率"), args.max_new_tokens)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_model": str(Path(args.base_model).resolve()),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "prompt_variant": args.prompt_variant,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "quantization": "4bit nf4 double-quant",
            "thinking": False,
            "repeats": args.repeats,
        },
        "splits": {},
    }
    for split in selected:
        cases = json.loads(SPLIT_FILES[split].read_text(encoding="utf-8"))
        if args.limit:
            cases = cases[:args.limit]
        payload["splits"][split] = evaluate_split(
            tokenizer, model, split, cases, args.prompt_variant, args.repeats, args.max_new_tokens
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "label": args.label,
        "output": str(output),
        "summaries": {name: value["summary"] for name, value in payload["splits"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
