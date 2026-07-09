"""
Prepare v3 DPO data.

v3 focuses on the seven repeated hard eval misses while avoiding exact overlap
with both hard_cases.json and hard_holdout_v3.json.
"""

import json
import random
import sys
from pathlib import Path

from prepare_dataset import SYSTEM_PROMPT
from prepare_dataset_v2 import DATA_DIR, dataset_entry, load_json_list

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DPO_SOURCE = DATA_DIR / "raw" / "dpo_boundary_v3.json"


def assert_no_eval_overlap(items: list[dict], eval_path: Path) -> None:
    cases = load_json_list(eval_path)
    eval_inputs = {case.get("input", "").strip() for case in cases}
    instructions = {item.get("instruction", "").strip() for item in items}
    overlap = sorted(instructions & eval_inputs)
    if overlap:
        joined = "\n".join(f"  - {text}" for text in overlap)
        raise ValueError(f"DPO v3 data overlaps with {eval_path.name}:\n{joined}")


def make_dpo_item(item: dict) -> dict:
    instruction = item["instruction"].strip()
    human_value = f"{SYSTEM_PROMPT}\n\n任务: {instruction}"
    return {
        "conversations": [{"from": "human", "value": human_value}],
        "chosen": {"from": "gpt", "value": json.dumps(item["chosen"], ensure_ascii=False)},
        "rejected": {"from": "gpt", "value": json.dumps(item["rejected"], ensure_ascii=False)},
    }


def merge_dataset_info(train_path: Path, eval_path: Path) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    info["tool_calling_dpo_train_v3"] = dataset_entry(train_path, ranking=True)
    info["tool_calling_dpo_eval_v3"] = dataset_entry(eval_path, ranking=True)

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main() -> None:
    random.seed(42)
    print("\nPreparing v3 DPO data")
    print("=" * 50)

    pairs = load_json_list(DPO_SOURCE)
    assert_no_eval_overlap(pairs, DATA_DIR / "eval" / "hard_cases.json")
    assert_no_eval_overlap(pairs, DATA_DIR / "eval" / "hard_holdout_v3.json")

    dpo_data = [make_dpo_item(item) for item in pairs]
    random.shuffle(dpo_data)

    split_idx = max(1, int(len(dpo_data) * 0.9))
    train_data = dpo_data[:split_idx]
    eval_data = dpo_data[split_idx:]

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "dpo_train_v3.json"
    eval_path = processed_dir / "dpo_eval_v3.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    merge_dataset_info(train_path, eval_path)

    print("\nDone")
    print(f"  pairs: {len(dpo_data)}")
    print(f"  train: {len(train_data)} -> {train_path}")
    print(f"  eval: {len(eval_data)} -> {eval_path}")


if __name__ == "__main__":
    main()
