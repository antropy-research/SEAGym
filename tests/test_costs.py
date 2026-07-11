from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seagym.costs import extract_token_cost, extract_token_cost_from_json_file


class CostExtractionTest(unittest.TestCase):
    def test_extracts_direct_total_tokens_without_estimating_missing_usage(self) -> None:
        self.assertEqual(extract_token_cost({"total_tokens": 123}), {"total_tokens": 123.0})
        self.assertEqual(extract_token_cost({"total_tokens": "N/A"}), {})
        self.assertEqual(extract_token_cost({"message": "no usage here"}), {})

    def test_extracts_provider_usage_with_parts(self) -> None:
        cost = extract_token_cost(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "cache_read_tokens": 2,
                    "total_tokens": 16,
                }
            }
        )

        self.assertEqual(
            cost,
            {
                "total_tokens": 16.0,
                "input_tokens": 10.0,
                "output_tokens": 4.0,
                "cache_tokens": 2.0,
            },
        )

    def test_sums_nested_usage_when_no_parent_total_exists(self) -> None:
        cost = extract_token_cost(
            {
                "steps": [
                    {"usage": {"total_tokens": 3}},
                    {"raw": {"usage": {"totalTokens": 7}}},
                ]
            }
        )

        self.assertEqual(cost, {"total_tokens": 10.0})

    def test_prefers_parent_total_to_avoid_double_counting_nested_usage(self) -> None:
        cost = extract_token_cost(
            {
                "total_tokens": 20,
                "steps": [
                    {"usage": {"total_tokens": 20}},
                    {"usage": {"total_tokens": 20}},
                ],
            }
        )

        self.assertEqual(cost, {"total_tokens": 20.0})

    def test_extracts_from_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"usage": {"total_tokens": 42}}), encoding="utf-8")

            self.assertEqual(extract_token_cost_from_json_file(path), {"total_tokens": 42.0})


if __name__ == "__main__":
    unittest.main()
