from __future__ import annotations

import unittest

from seagym.metrics import MetricRegistry, default_metric_registry


class MetricsTest(unittest.TestCase):
    def test_success_rate_by_view(self) -> None:
        records = [
            {"view_name": "id_test", "success": True, "score": 1, "attributes": {"domain": "code"}},
            {"view_name": "id_test", "success": False, "score": 0, "attributes": {"domain": "code"}},
            {"view_name": "ood_test", "success": True, "score": 1, "attributes": {"domain": "data_workflow"}},
        ]

        metrics = default_metric_registry().compute(records, {}, ["success_rate", "mean_score"])

        self.assertEqual(metrics["success_rate"]["id_test"], 0.5)
        self.assertEqual(metrics["success_rate"]["ood_test"], 1.0)
        self.assertEqual(metrics["mean_score"]["id_test"], 0.5)

    def test_custom_metric_import_path(self) -> None:
        registry = MetricRegistry.from_config(
            {
                "registry": [
                    {
                        "name": "custom_alias",
                        "type": "python",
                        "import_path": "tests.custom_metric_fixture:ConstantMetric",
                    }
                ]
            }
        )

        metrics = registry.compute([{"view_name": "x", "success": True, "score": 1}], {}, ["custom_alias"])

        self.assertEqual(metrics["custom_alias"], {"value": 7, "num_records": 1})

    def test_protocol_metrics_use_evaluation_points_and_final_roles(self) -> None:
        records = [
            {"view_name": "update_validation", "evaluation_point_id": "E_0", "success": True, "score": 0.5},
            {"view_name": "update_validation", "evaluation_point_id": "E_1", "success": True, "score": 0.75},
            {"view_name": "update_validation", "evaluation_point_id": "E_2", "success": True, "score": 0.5},
            {"view_name": "id_test", "baseline_role": "A_0", "success": False, "score": 0.25},
            {"view_name": "id_test", "baseline_role": "A_T", "success": True, "score": 0.75},
        ]

        metrics = default_metric_registry().compute(
            records,
            {"update_assessment": {"min_improvement": 0.1}},
            [
                "success_rate",
                "update_validation_gain",
                "validation_supported_update_rate",
                "final_gain",
                "forgetting_rate",
            ],
        )

        self.assertEqual(metrics["success_rate"]["id_test.A_0"], 0.0)
        self.assertEqual(metrics["success_rate"]["id_test.A_T"], 1.0)
        self.assertEqual(metrics["update_validation_gain"]["prev"], {"E_1": 0.25, "E_2": -0.25})
        self.assertEqual(metrics["update_validation_gain"]["base"], {"E_1": 0.25, "E_2": 0.0})
        self.assertEqual(metrics["validation_supported_update_rate"]["value"], 0.5)
        self.assertEqual(metrics["final_gain"]["id_test"], 0.5)
        self.assertEqual(metrics["forgetting_rate"]["id_test"], 0.0)

    def test_token_usage_splits_rollout_and_update_costs(self) -> None:
        records = [
            {
                "view_name": "train",
                "mode": "train",
                "cost": {"n_input_tokens": 10, "n_cache_tokens": 2, "n_output_tokens": 3, "cost_usd": 0.10},
            },
            {
                "view_name": "id_test",
                "mode": "final",
                "cost": {"tokens": 20, "cost_usd": 0.20},
            },
            {
                "view_name": "agent_update",
                "mode": "update",
                "cost": {"input_tokens": 5, "output_tokens": 7, "cost_usd": 0.30},
            },
        ]

        metrics = default_metric_registry().compute(records, {}, ["tokens", "cost_usd"])

        self.assertEqual(metrics["tokens"]["rollout"]["total_tokens"], 35.0)
        self.assertEqual(metrics["tokens"]["rollout"]["mean_tokens"], 17.5)
        self.assertEqual(metrics["tokens"]["update"]["total_tokens"], 12.0)
        self.assertEqual(metrics["tokens"]["update"]["mean_tokens"], 12.0)
        self.assertEqual(metrics["tokens"]["overall"]["total_tokens"], 47.0)
        self.assertAlmostEqual(metrics["cost_usd"]["overall"]["total"], 0.60)


if __name__ == "__main__":
    unittest.main()
