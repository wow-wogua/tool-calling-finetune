"""
Prepare the v3 SFT dataset.

v3 treats data/eval/hard_cases.json as a dev set that exposed failure modes.
It adds boundary_v3.json with similar-but-not-identical samples and keeps
hard_holdout_v3.json for a fresh validation pass.
"""

import json
import random
import sys
from pathlib import Path

from prepare_dataset import convert_to_sft, parse_output, split_dataset
from prepare_dataset_v2 import DATA_DIR, dataset_entry, get_instruction, load_json_list

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent

RAW_FILES = [
    ("handcrafted", DATA_DIR / "raw" / "handcrafted.json"),
    ("generated_v2", DATA_DIR / "raw" / "generated_v2.json"),
    ("boundary_v2", DATA_DIR / "raw" / "boundary_v2.json"),
    ("boundary_v3", DATA_DIR / "raw" / "boundary_v3.json"),
]


def assert_no_eval_overlap(items: list[dict], source_name: str, eval_path: Path) -> None:
    cases = load_json_list(eval_path)
    eval_inputs = {case.get("input", "").strip() for case in cases}
    instructions = {get_instruction(item) for item in items if get_instruction(item)}
    overlap = sorted(instructions & eval_inputs)
    if overlap:
        joined = "\n".join(f"  - {text}" for text in overlap)
        raise ValueError(f"{source_name} overlaps with {eval_path.name}:\n{joined}")


def load_raw_data() -> list[dict]:
    all_data: list[dict] = []
    hard_eval = DATA_DIR / "eval" / "hard_cases.json"
    holdout_eval = DATA_DIR / "eval" / "hard_holdout_v3.json"

    for name, path in RAW_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        data = load_json_list(path)
        assert_no_eval_overlap(data, name, hard_eval)
        assert_no_eval_overlap(data, name, holdout_eval)
        all_data.extend(data)
        print(f"  loaded {name}: {len(data)} rows")

    seen: set[str] = set()
    unique: list[dict] = []
    duplicates = 0
    for item in all_data:
        instruction = get_instruction(item)
        if not instruction:
            continue
        if instruction in seen:
            duplicates += 1
            continue
        seen.add(instruction)
        unique.append(item)

    if duplicates:
        print(f"  deduplicated by instruction: removed {duplicates}, kept {len(unique)}")
    return unique


def merge_dataset_info(train_path: Path, eval_path: Path) -> None:
    info_path = DATA_DIR / "dataset_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    info["tool_calling_train_v3"] = dataset_entry(train_path)
    info["tool_calling_eval_v3"] = dataset_entry(eval_path)

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main() -> None:
    random.seed(42)
    print("\nPreparing v3 SFT data")
    print("=" * 50)

    raw_data = load_raw_data()
    tool_counts: dict[str, int] = {}
    for item in raw_data:
        output = parse_output(item.get("output", ""))
        tool = output.get("tool", "unknown") if isinstance(output, dict) else "unknown"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\nTool distribution:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")

    train_raw, eval_raw = split_dataset(raw_data, eval_ratio=0.2, seed=42)
    train_sft = [convert_to_sft(item) for item in train_raw]
    eval_sft = [convert_to_sft(item) for item in eval_raw]

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train_v3.json"
    eval_path = processed_dir / "eval_v3.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_sft, f, ensure_ascii=False, indent=2)

    merge_dataset_info(train_path, eval_path)

    print("\nDone")
    print(f"  raw unique: {len(raw_data)}")
    print(f"  train: {len(train_sft)} -> {train_path}")
    print(f"  eval: {len(eval_sft)} -> {eval_path}")


if __name__ == "__main__":
    main()
