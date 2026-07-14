from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_three_models import relationship


def summaries(hard: float, holdout: float, capability: float) -> dict:
    return {
        "hard": {"full_accuracy": hard},
        "holdout": {"full_accuracy": holdout},
        "capability_holdout": {"full_accuracy": capability},
    }


class RelationshipTests(unittest.TestCase):
    def test_exceeds_requires_no_worse_split(self) -> None:
        result = relationship(summaries(95, 96, 88), summaries(90, 70, 88))
        self.assertEqual(result["classification"], "exceeds_on_frozen_narrow_eval")

    def test_close_when_all_splits_within_five_points(self) -> None:
        result = relationship(summaries(90, 90, 90), summaries(92, 87, 94))
        self.assertEqual(result["classification"], "close_on_frozen_narrow_eval")

    def test_below_when_one_split_is_materially_worse(self) -> None:
        result = relationship(summaries(95, 70, 90), summaries(90, 80, 90))
        self.assertEqual(result["classification"], "below_on_frozen_narrow_eval")


if __name__ == "__main__":
    unittest.main()
