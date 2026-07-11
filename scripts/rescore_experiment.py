"""不重新推理，仅用当前归一化规则重算已有公平实验结果。"""

import argparse
import json
from pathlib import Path

from evaluation_core import score_tool_call, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_rescored.json")
    experiment = json.loads(input_path.read_text(encoding="utf-8"))

    for result in experiment["results"].values():
        for detail in result["details"]:
            actual = {
                "tool": detail.get("actual_tool"),
                "params": detail.get("actual_params", {}),
            }
            expected_params = detail.get("normalized_expected_params", {})
            detail.update(score_tool_call(actual, detail.get("expected_tool"), expected_params, detail.get("input", "")))
        metrics = summarize(result["details"])
        result.update(metrics)

    experiment["rescored_from"] = str(input_path)
    output_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
