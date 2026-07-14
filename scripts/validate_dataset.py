"""Validate v4 schema, frozen hashes, distributions, and leakage guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from researcher_prompt_v4 import build_researcher_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
EVAL_ROOT = DATA_ROOT / "eval"
V4_ROOT = DATA_ROOT / "v4"
LOCK_PATH = V4_ROOT / "manifests" / "holdout_lock.json"
MANIFEST_PATH = V4_ROOT / "manifests" / "v4_manifest.json"
REPORT_PATH = V4_ROOT / "manifests" / "validation_report.json"
ALLOWED_TOOLS = {"search_videos", "rag_search", "get_transcript", "get_trend_data", "none"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (value or "").lower())


def validate_case(item: dict, location: str, errors: list[str]) -> None:
    required = {
        "id", "input", "platforms", "capability_profile", "capabilities",
        "expected_tool", "expected_params", "intent_family", "expression_family",
        "parameter_pattern", "source",
    }
    missing = sorted(required - set(item))
    if missing:
        errors.append(f"{location}: missing fields {missing}")
        return
    tool = item["expected_tool"]
    params = item["expected_params"]
    if tool not in ALLOWED_TOOLS:
        errors.append(f"{location}: invalid expected_tool={tool}")
        return
    if not isinstance(params, dict):
        errors.append(f"{location}: expected_params must be an object")
        return
    if tool == "none" and params:
        errors.append(f"{location}: none must have empty params")
    if tool != "none":
        state = item["capabilities"].get(tool, {})
        if not state.get("enabled"):
            errors.append(f"{location}: expected tool {tool} is disabled")
    if tool == "search_videos":
        platforms = params.get("platforms", ["bilibili"])
        if platforms != ["bilibili"]:
            errors.append(f"{location}: search_videos only supports bilibili")
        limit = params.get("limit")
        if limit is not None and not (isinstance(limit, int) and 1 <= limit <= 20):
            errors.append(f"{location}: invalid search limit={limit}")
    if tool == "rag_search":
        if not isinstance(params.get("query"), str) or not params["query"].strip():
            errors.append(f"{location}: rag_search requires query")
        top_k = params.get("top_k")
        if top_k is not None and not (isinstance(top_k, int) and 1 <= top_k <= 10):
            errors.append(f"{location}: invalid top_k={top_k}")
    if tool == "get_transcript":
        url = params.get("video_url", "")
        if not isinstance(url, str) or url not in item["input"]:
            errors.append(f"{location}: transcript URL must appear in input")
        if not (url.startswith("https://www.bilibili.com/") or url.startswith("https://b23.tv/")):
            errors.append(f"{location}: transcript URL must be a public bilibili URL")
    if tool == "get_trend_data":
        video_id = params.get("video_id", "")
        if not video_id or video_id not in item["input"]:
            errors.append(f"{location}: trend video_id must appear in input")


def verify_frozen_eval(errors: list[str]) -> tuple[dict, dict[str, list[dict]]]:
    if not LOCK_PATH.exists():
        errors.append("holdout lock is missing")
        return {}, {}
    lock = load_json(LOCK_PATH)
    datasets = {}
    observed_hashes = {}
    ids = set()
    for filename, expected in lock.get("files", {}).items():
        path = EVAL_ROOT / filename
        if not path.exists():
            errors.append(f"frozen eval file missing: {filename}")
            continue
        actual_hash = sha256_file(path)
        observed_hashes[filename] = actual_hash
        if actual_hash != expected.get("sha256"):
            errors.append(f"frozen hash mismatch: {filename}")
        cases = load_json(path)
        datasets[filename] = cases
        if len(cases) != expected.get("count"):
            errors.append(f"frozen count mismatch: {filename}")
        for index, item in enumerate(cases):
            validate_case(item, f"{filename}[{index}]", errors)
            if item.get("id") in ids:
                errors.append(f"duplicate eval id: {item.get('id')}")
            ids.add(item.get("id"))
    combined = hashlib.sha256(
        json.dumps(observed_hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if combined != lock.get("combined_sha256"):
        errors.append("combined frozen eval hash mismatch")
    return lock, datasets


def validate_training(
    eval_datasets: dict[str, list[dict]], errors: list[str], warnings: list[str]
) -> dict:
    paths = {
        "raw": V4_ROOT / "raw" / "cases.json",
        "train_cases": V4_ROOT / "processed" / "train_cases.json",
        "validation_cases": V4_ROOT / "processed" / "validation_cases.json",
        "train": V4_ROOT / "processed" / "train.json",
        "validation": V4_ROOT / "processed" / "validation.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        errors.append(f"training files missing: {missing}")
        return {}
    raw = load_json(paths["raw"])
    train_cases = load_json(paths["train_cases"])
    validation_cases = load_json(paths["validation_cases"])
    train_rows = load_json(paths["train"])
    validation_rows = load_json(paths["validation"])

    all_ids = [item.get("id") for item in raw]
    if len(all_ids) != len(set(all_ids)):
        errors.append("duplicate training case ids")
    for index, item in enumerate(raw):
        validate_case(item, f"raw[{index}]", errors)
        if item.get("source") in {"mimo_generated", "deepseek_generated", "llm_generated", "online_user_data"}:
            errors.append(f"forbidden source label: {item.get('source')}")

    split_ids = {item["id"] for item in train_cases} | {item["id"] for item in validation_cases}
    if split_ids != set(all_ids):
        errors.append("train/validation metadata is not an exact partition of raw cases")
    if {item["id"] for item in train_cases} & {item["id"] for item in validation_cases}:
        errors.append("train and validation ids overlap")
    if len(train_rows) != len(train_cases) or len(validation_rows) != len(validation_cases):
        errors.append("ShareGPT row counts do not match case metadata")

    for metadata, rows, label in (
        (train_cases, train_rows, "train"),
        (validation_cases, validation_rows, "validation"),
    ):
        for index, (item, row) in enumerate(zip(metadata, rows)):
            expected_prompt = build_researcher_prompt(
                item["input"], item["platforms"], item["capabilities"], variant="contract"
            )
            conversations = row.get("conversations", [])
            if len(conversations) != 2:
                errors.append(f"{label}[{index}]: expected two ShareGPT messages")
                continue
            if conversations[0].get("from") != "human" or conversations[0].get("value") != expected_prompt:
                errors.append(f"{label}[{index}]: prompt mismatch")
            try:
                answer = json.loads(conversations[1].get("value", ""))
            except json.JSONDecodeError:
                errors.append(f"{label}[{index}]: assistant answer is not JSON")
                continue
            expected = {"tool": item["expected_tool"], "params": item["expected_params"]}
            if answer != expected:
                errors.append(f"{label}[{index}]: assistant answer mismatch")

    frozen_cases = [item for cases in eval_datasets.values() for item in cases]
    frozen_families = {item["expression_family"] for item in frozen_cases}
    train_families = {item["expression_family"] for item in raw}
    family_overlap = sorted(frozen_families & train_families)
    if family_overlap:
        errors.append(f"expression family leakage: {family_overlap}")

    frozen_normalized = {}
    for item in frozen_cases:
        frozen_normalized.setdefault(normalize_text(item["input"]), []).append(item["id"])
    exact_overlap = []
    for item in raw:
        normalized = normalize_text(item["input"])
        if normalized in frozen_normalized:
            exact_overlap.append({"train": item["id"], "eval": frozen_normalized[normalized]})
    if exact_overlap:
        errors.append(f"exact or normalized text leakage: {exact_overlap[:10]}")

    near_overlap = []
    frozen_texts = [(item["id"], normalize_text(item["input"])) for item in frozen_cases]
    for item in raw:
        normalized = normalize_text(item["input"])
        if len(normalized) < 12:
            continue
        for eval_id, eval_text in frozen_texts:
            if len(eval_text) < 12:
                continue
            length_ratio = min(len(normalized), len(eval_text)) / max(len(normalized), len(eval_text))
            if length_ratio < 0.78:
                continue
            ratio = SequenceMatcher(None, normalized, eval_text).ratio()
            if ratio >= 0.96:
                near_overlap.append({"train": item["id"], "eval": eval_id, "ratio": round(ratio, 4)})
    if near_overlap:
        errors.append(f"near-duplicate leakage: {near_overlap[:10]}")

    if not MANIFEST_PATH.exists():
        errors.append("v4 manifest is missing")
    else:
        manifest = load_json(MANIFEST_PATH)
        for name, info in manifest.get("files", {}).items():
            path = DATA_ROOT / info["file_name"]
            if not path.exists() or sha256_file(path) != info.get("sha256"):
                errors.append(f"manifest hash mismatch: {name}")
        truthful_sources = set(manifest.get("truthful_sources", []))
        observed_sources = {item["source"] for item in raw}
        if observed_sources - truthful_sources:
            errors.append(f"manifest omits sources: {sorted(observed_sources - truthful_sources)}")

    return {
        "raw_count": len(raw),
        "train_count": len(train_cases),
        "validation_count": len(validation_cases),
        "tool_distribution": dict(sorted(Counter(item["expected_tool"] for item in raw).items())),
        "source_distribution": dict(sorted(Counter(item["source"] for item in raw).items())),
        "family_overlap_count": len(family_overlap),
        "exact_overlap_count": len(exact_overlap),
        "near_overlap_count": len(near_overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    lock, eval_datasets = verify_frozen_eval(errors)
    training_summary = {} if args.eval_only else validate_training(eval_datasets, errors, warnings)
    report = {
        "valid": not errors,
        "mode": "eval-only" if args.eval_only else "full",
        "frozen_combined_sha256": lock.get("combined_sha256"),
        "eval_counts": {name: len(cases) for name, cases in eval_datasets.items()},
        "training": training_summary,
        "errors": errors,
        "warnings": warnings,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
