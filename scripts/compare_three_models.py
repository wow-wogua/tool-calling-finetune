"""Build the frozen three-model comparison without making API requests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluate_v4 import SPLIT_FILES, score_case, sha256_file, strict_json, summarize, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "results" / "v4" / "baseline_base_contract_rescored.json"
DEFAULT_ADAPTER = PROJECT_ROOT / "results" / "v4" / "v4_1_final_repeated.json"
DEFAULT_DEEPSEEK = PROJECT_ROOT / "results" / "v4" / "deepseek_v4_pro_frozen_eval.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "v4" / "final_three_model_comparison.json"
SPLITS = ("hard", "holdout", "capability_holdout")
METRICS = (
    "tool_accuracy",
    "params_accuracy",
    "full_accuracy",
    "safe_accuracy",
    "json_valid_rate",
    "unavailable_tool_violation_rate",
    "unsupported_platform_violation_rate",
    "hallucinated_id_or_url_rate",
    "capability_state_accuracy",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def select_unique_details(payload: dict, split: str) -> list[dict]:
    details = payload["splits"][split]["details"]
    by_id = {}
    for detail in details:
        if detail.get("repeat", 1) == 1:
            by_id[detail["id"]] = detail
    expected_ids = [item["id"] for item in load_json(SPLIT_FILES[split])]
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise ValueError(f"{split} case mismatch: missing={missing}, extra={extra}")
    return [by_id[case_id] for case_id in expected_ids]


def rescore_details(payload: dict, split: str) -> tuple[list[dict], list[str]]:
    cases = {item["id"]: item for item in load_json(SPLIT_FILES[split])}
    rescored = []
    mismatches = []
    for old in select_unique_details(payload, split):
        case = cases[old["id"]]
        actual, json_valid = strict_json(old["raw_output"])
        scored = score_case(case, actual, json_valid)
        for field in (
            "actual_tool",
            "actual_params",
            "tool_correct",
            "params_correct",
            "fully_correct",
            "safe_correct",
            "json_valid",
            "unavailable_tool_violation",
            "unsupported_platform_violation",
            "hallucinated_id_or_url",
            "capability_correct",
            "error_types",
        ):
            if old.get(field) != scored[field]:
                mismatches.append(f"{split}:{old['id']}:{field}")
        rescored.append({**old, **scored})
    return rescored, mismatches


def aggregate(model_details: dict[str, list[dict]]) -> dict:
    details = [item for split in SPLITS for item in model_details[split]]
    summary = summarize(details)
    summary["latency_ms"] = {
        "mean": summary["latency_ms"]["mean"],
        "p50": summary["latency_ms"]["p50"],
        "p95": summary["latency_ms"]["p95"],
    }
    return summary


def relationship(adapter_summaries: dict, deepseek_summaries: dict) -> dict:
    differences = {
        split: adapter_summaries[split]["full_accuracy"]
        - deepseek_summaries[split]["full_accuracy"]
        for split in SPLITS
    }
    if all(value >= 0 for value in differences.values()) and any(
        value > 0 for value in differences.values()
    ):
        classification = "exceeds_on_frozen_narrow_eval"
        statement = "在冻结的自建窄域Researcher评测上超过DeepSeek V4 Pro。"
        decision = {
            "model_layer": "v4.1 becomes the preferred Researcher candidate for further validation",
            "project2_integration": "keep DeepSeek as the project-2 default; do not switch automatically",
            "unverified": [
                "broader end-to-end task quality",
                "long-running service stability",
                "online production generalization",
                "strict infrastructure-normalized latency and cost",
            ],
            "evidence_that_would_change_decision": [
                "a broader frozen project-2 end-to-end task set",
                "long-duration service reliability results",
                "production-like monitoring showing stable quality and failure rates",
            ],
        }
    elif all(abs(value) <= 5.0 for value in differences.values()):
        classification = "close_on_frozen_narrow_eval"
        statement = "4B本地专用模型在冻结窄域工具路由评测上接近DeepSeek V4 Pro。"
        decision = {
            "model_layer": "retain v4.1 as a localization, private-deployment, or provider-fallback candidate",
            "project2_integration": "keep DeepSeek as the project-2 default",
            "unverified": [
                "broader end-to-end task quality",
                "long-running service stability",
                "online production generalization",
            ],
            "evidence_that_would_change_decision": [
                "a broader frozen project-2 end-to-end task set",
                "reliability and infrastructure-normalized cost measurements",
            ],
        }
    else:
        classification = "below_on_frozen_narrow_eval"
        statement = "微调显著改善Qwen3-4B基座，但尚不能替代DeepSeek V4 Pro。"
        decision = {
            "model_layer": "v4.1 does not meet the production replacement threshold",
            "project2_integration": "keep DeepSeek as the project-2 default and do not add v4.1 to the default path",
            "unverified": ["online production generalization"],
            "evidence_that_would_change_decision": [
                "a new project requirement that changes the frozen contract",
                "a future independently frozen evaluation showing the gap has closed",
            ],
        }
    return {
        "classification_rule": "exceeds only when v4.1 is no worse on every frozen split and better on at least one; close requires every split within 5 percentage points; otherwise below",
        "full_accuracy_difference_pp_v4_1_minus_deepseek": differences,
        "classification": classification,
        "allowed_statement": statement,
        "engineering_decision": decision,
    }


def build_comparison(base_path: Path, adapter_path: Path, deepseek_path: Path) -> dict:
    sources = {
        "qwen_base_contract": load_json(base_path),
        "qwen_v4_1_direct_adapter": load_json(adapter_path),
        "deepseek_v4_pro": load_json(deepseek_path),
    }
    details = {}
    consistency_mismatches = []
    summaries = {}
    for model_name, payload in sources.items():
        details[model_name] = {}
        summaries[model_name] = {}
        for split in SPLITS:
            rescored, mismatches = rescore_details(payload, split)
            details[model_name][split] = rescored
            summaries[model_name][split] = summarize(rescored)
            consistency_mismatches.extend(f"{model_name}:{item}" for item in mismatches)

    per_split = {}
    for split in SPLITS:
        per_split[split] = {
            "case_count": len(details["deepseek_v4_pro"][split]),
            "metrics": {
                metric: {
                    model_name: summaries[model_name][split][metric]
                    for model_name in sources
                }
                for metric in METRICS
            },
            "latency_ms": {
                model_name: summaries[model_name][split]["latency_ms"]
                for model_name in sources
            },
            "error_types": {
                model_name: summaries[model_name][split]["error_types"]
                for model_name in sources
            },
        }

    overall = {model_name: aggregate(model_details) for model_name, model_details in details.items()}
    deepseek_usage = sources["deepseek_v4_pro"].get("usage", {})
    rel = relationship(
        summaries["qwen_v4_1_direct_adapter"], summaries["deepseek_v4_pro"]
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen project-2 Researcher tool-routing contract",
        "frozen_eval": {
            "counts": {split: len(load_json(SPLIT_FILES[split])) for split in SPLITS},
            "hashes": {split: sha256_file(SPLIT_FILES[split]) for split in SPLITS},
            "total": sum(len(load_json(SPLIT_FILES[split])) for split in SPLITS),
        },
        "source_files": {
            "qwen_base_contract": project_relative(base_path),
            "qwen_v4_1_direct_adapter": project_relative(adapter_path),
            "deepseek_v4_pro": project_relative(deepseek_path),
        },
        "fairness": {
            "same_prompt_variant": "contract",
            "same_case_ids": True,
            "same_capability_state": True,
            "same_tool_schema": True,
            "same_scoring_core": True,
            "qwen_base_vs_v4_1_meaning": "isolates the effect of the Direct Adapter on the same Qwen3-4B base",
            "v4_1_vs_deepseek_meaning": "estimates narrow-domain Researcher candidate value, not general model superiority",
            "project2_three_case_ab_meaning": "validates real Agent-path integration only",
        },
        "scoring_consistency": {
            "passed": not consistency_mismatches,
            "mismatches": consistency_mismatches,
        },
        "per_split": per_split,
        "overall": {model: {metric: summary[metric] for metric in METRICS} for model, summary in overall.items()},
        "overall_latency_ms": {model: summary["latency_ms"] for model, summary in overall.items()},
        "deepseek_api_usage": deepseek_usage,
        "cost_and_latency_notes": [
            "DeepSeek cost uses the project-2 configured estimate of USD 0.435/M input tokens and USD 0.87/M output tokens.",
            "Qwen latency is local RTX 4060 Laptop GPU inference; DeepSeek latency is remote API latency, so latency is descriptive rather than infrastructure-normalized.",
            "The experiment does not establish a full local serving cost or production total cost of ownership.",
        ],
        "relationship": rel,
        "error_analysis": {
            "deepseek": {
                "error_types": dict(sorted(Counter(
                    error
                    for split in SPLITS
                    for item in details["deepseek_v4_pro"][split]
                    for error in item["error_types"]
                ).items())),
                "notable": "DeepSeek was strongest on parameter accuracy but had more wrong-tool errors on holdout plus one unsupported-platform call and one hallucinated URL/ID event.",
            },
            "qwen_v4_1": {
                "error_types": dict(sorted(Counter(
                    error
                    for split in SPLITS
                    for item in details["qwen_v4_1_direct_adapter"][split]
                    for error in item["error_types"]
                ).items())),
                "notable": "v4.1 retained zero unavailable-tool, unsupported-platform, and URL/ID hallucination violations, but capability-holdout parameter accuracy remained weaker than DeepSeek.",
            },
        },
        "claim_boundaries": {
            "allowed": [
                rel["allowed_statement"],
                "Qwen Base vs v4.1 demonstrates that the LoRA Direct Adapter materially improved the frozen routing task.",
            ],
            "not_allowed": [
                "v4.1 comprehensively outperforms DeepSeek",
                "project 2 has switched its default Researcher model",
                "the offline result proves online product superiority",
                "local inference is already proven cheaper or faster in production",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--deepseek", default=str(DEFAULT_DEEPSEEK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    comparison = build_comparison(Path(args.base), Path(args.adapter), Path(args.deepseek))
    if not comparison["scoring_consistency"]["passed"]:
        raise SystemExit("scoring consistency check failed")
    write_json_atomic(Path(args.output), comparison)
    print(json.dumps({
        "output": args.output,
        "relationship": comparison["relationship"],
        "overall": comparison["overall"],
        "deepseek_api_usage": comparison["deepseek_api_usage"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
