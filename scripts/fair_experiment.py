"""在同一 hard/holdout 用例与生成设置下运行 5 组公平实验。"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from evaluation_core import parse_model_output, score_tool_call, summarize
from researcher_prompt import build_researcher_prompt


PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "Qwen3-4B-base",
    Path.home() / ".cache" / "modelscope" / "Qwen" / "Qwen3-4B",
    Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-4B",
)
DEFAULT_BASE_MODEL = next((path for path in BASE_MODEL_CANDIDATES if (path / "config.json").exists()), BASE_MODEL_CANDIDATES[0])
DEFAULT_FINETUNED_MODEL = PROJECT_ROOT / "outputs" / "qwen3_dpo_tool_calling_merged_v3"


def load_cases(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"评测用例为空: {path}")
    return cases


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization,
    )
    model.eval()
    return tokenizer, model


def generate_once(tokenizer, model, prompt: str) -> tuple[str, float]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
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
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), latency_ms


def warm_up(tokenizer, model):
    generate_once(tokenizer, model, build_researcher_prompt("你好，不需要调用工具", "base"))


def evaluate_dataset(tokenizer, model, cases: list[dict], prompt_variant: str) -> dict:
    results = []
    for index, case in enumerate(cases, start=1):
        prompt = build_researcher_prompt(case["input"], prompt_variant)
        response, latency_ms = generate_once(tokenizer, model, prompt)
        actual, json_valid = parse_model_output(response)
        score = score_tool_call(actual, case.get("expected_tool"), case.get("expected_params", {}), case["input"])
        result = {
            "case": index,
            "input": case["input"],
            "expected_tool": case.get("expected_tool"),
            "actual_tool": actual.get("tool"),
            "actual_params": actual.get("params", {}),
            "json_valid": json_valid,
            "latency_ms": round(latency_ms, 2),
            **score,
        }
        results.append(result)
        status = "✅" if result["fully_correct"] else ("⚠️" if result["tool_correct"] else "❌")
        print(
            f"  {status} [{index:02d}/{len(cases)}] {case['input'][:24]} "
            f"期望={case.get('expected_tool')} 实际={actual.get('tool')} {latency_ms:.0f}ms"
        )
    return {**summarize(results), "prompt_variant": prompt_variant, "details": results}


def release_model(model, tokenizer):
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model_matrix(model_label: str, model_path: str, variants: list[str], datasets: dict) -> dict:
    tokenizer, model = load_model(model_path)
    warm_up(tokenizer, model)
    output = {}
    for variant in variants:
        for dataset_name, cases in datasets.items():
            key = f"{model_label}+{variant}+{dataset_name}"
            print(f"\n{'=' * 70}\n{key}\n{'=' * 70}")
            output[key] = {
                "model_label": model_label,
                "model_path": model_path,
                "dataset": dataset_name,
                **evaluate_dataset(tokenizer, model, cases, variant),
            }
    release_model(model, tokenizer)
    return output


def main():
    parser = argparse.ArgumentParser(description="项目三 5 组公平实验")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--finetuned-model", default=str(DEFAULT_FINETUNED_MODEL))
    parser.add_argument("--hard-cases", default=str(PROJECT_ROOT / "data" / "eval" / "hard_cases.json"))
    parser.add_argument("--holdout-cases", default=str(PROJECT_ROOT / "data" / "eval" / "hard_holdout_v3.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / f"fair_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    args = parser.parse_args()

    datasets = {
        "hard44": load_cases(Path(args.hard_cases)),
        "holdout20": load_cases(Path(args.holdout_cases)),
    }
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "generation": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 128, "quantization": "4bit"},
        "datasets": {name: len(cases) for name, cases in datasets.items()},
        "matrix": [
            "base+base",
            "base+strengthened",
            "base+rules",
            "sft_dpo_v3+base",
            "sft_dpo_v3+rules",
        ],
        "results": {},
    }
    experiment["results"].update(run_model_matrix("base", args.base_model, ["base", "strengthened", "rules"], datasets))
    experiment["results"].update(run_model_matrix("sft_dpo_v3", args.finetuned_model, ["base", "rules"], datasets))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
