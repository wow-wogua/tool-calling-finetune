from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_v4


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"tool":"none","params":{}}'
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=123,
                completion_tokens=9,
                total_tokens=132,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class EvaluateV4ApiTests(unittest.TestCase):
    def test_api_generate_uses_frozen_generation_contract(self) -> None:
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.json"
            response, _, usage, attempts = evaluate_v4.api_generate(
                client=client,
                model="deepseek-v4-pro",
                prompt="frozen prompt",
                max_tokens=96,
                case_key="hard:hard-01",
                checkpoint={"failed_attempts": []},
                checkpoint_path=checkpoint_path,
                max_retries=2,
                retry_backoff_seconds=0.0,
            )
        self.assertEqual(response, '{"tool":"none","params":{}}')
        self.assertEqual(usage, {"input_tokens": 123, "output_tokens": 9, "total_tokens": 132})
        self.assertEqual(attempts, 1)
        self.assertEqual(client.completions.kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(client.completions.kwargs["temperature"], 0.0)
        self.assertEqual(client.completions.kwargs["top_p"], 1.0)
        self.assertEqual(client.completions.kwargs["max_tokens"], 96)
        self.assertEqual(
            client.completions.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(
            client.completions.kwargs["messages"],
            [{"role": "user", "content": "frozen prompt"}],
        )

    def test_checkpoint_rejects_changed_frozen_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": "deepseek-v4-pro",
                        "prompt_variant": "contract",
                        "max_tokens": 96,
                        "split_hashes": {"hard": "old-hash"},
                        "completed": {},
                        "failed_attempts": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                evaluate_v4.load_api_checkpoint(
                    path,
                    "deepseek-v4-pro",
                    "contract",
                    96,
                    {"hard": "new-hash"},
                )

    def test_usage_summary_uses_project2_pricing(self) -> None:
        summary = evaluate_v4.api_usage_summary(
            [
                {"usage": {"input_tokens": 1000, "output_tokens": 100}, "attempts": 1, "retries": 0},
                {"usage": {"input_tokens": 2000, "output_tokens": 200}, "attempts": 2, "retries": 1},
            ],
            input_cost_per_million=0.435,
            output_cost_per_million=0.87,
        )
        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["attempts"], 3)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["input_tokens"], 3000)
        self.assertEqual(summary["output_tokens"], 300)
        self.assertEqual(summary["estimated_cost_usd"], 0.001566)


if __name__ == "__main__":
    unittest.main()
