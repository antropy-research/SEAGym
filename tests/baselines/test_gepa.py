from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from seagym.baselines import build_baseline
from seagym.baselines.data import Trajectory, TrajectoryBatch
from seagym.baselines.gepa import GEPABaseline, _render_candidate
from seagym.envs import TaskRunResult


class GEPABaselineTest(unittest.TestCase):
    def test_gepa_update_requires_real_evaluator(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                built = build_baseline(
                    {
                        "baseline": {
                            "name": "gepa",
                            "class_path": "seagym.baselines.gepa:GEPABaseline",
                            "config": {
                                "project_dir": "reference/gepa",
                            },
                        }
                    },
                    run_dir=Path(tmp) / "gepa",
                    base_dir=Path.cwd(),
                )
                state = built.baseline.initialize(Path(tmp) / "run")
                empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

                with self.assertRaisesRegex(RuntimeError, "evaluator_import_path"):
                    built.baseline.update(empty_batch, state)

    def test_gepa_update_model_config_maps_deepseek_for_litellm(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                built = build_baseline(
                    {
                        "baseline": {
                            "name": "gepa",
                            "class_path": "seagym.baselines.gepa:GEPABaseline",
                            "config": {
                                "project_dir": "reference/gepa",
                                "update_model_ref": "update_model",
                            },
                            "models": {
                                "update_model": {
                                    "provider": "deepseek",
                                    "model": "deepseek/deepseek-v4-flash",
                                    "api_key_env": "DEEPSEEK_API_KEY",
                                }
                            },
                        }
                    },
                    run_dir=Path(tmp) / "gepa",
                    base_dir=Path.cwd(),
                )

                baseline = built.baseline
                self.assertIsInstance(baseline, GEPABaseline)
                assert isinstance(baseline, GEPABaseline)
                self.assertEqual(baseline.reflection_lm, "deepseek/deepseek-v4-flash")
                self.assertEqual(baseline.reflection_lm_api_key_env, "DEEPSEEK_API_KEY")

    def test_gepa_update_model_config_maps_openai_and_glm_for_litellm(self) -> None:
            old_key = os.environ.get("GLM_API_KEY")
            os.environ["GLM_API_KEY"] = "test-glm-key"
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    built = build_baseline(
                        {
                            "baseline": {
                                "name": "gepa",
                                "class_path": "seagym.baselines.gepa:GEPABaseline",
                                "config": {
                                    "project_dir": "reference/gepa",
                                    "update_model_ref": "update_model",
                                },
                                "models": {
                                    "update_model": {
                                        "provider": "openai_compatible",
                                        "model": "glm-4-plus",
                                        "api_base": "https://api.z.ai/api/coding/paas/v4",
                                        "api_key_env": "GLM_API_KEY",
                                    }
                                },
                            }
                        },
                        run_dir=Path(tmp) / "gepa",
                        base_dir=Path.cwd(),
                    )
            finally:
                if old_key is None:
                    os.environ.pop("GLM_API_KEY", None)
                else:
                    os.environ["GLM_API_KEY"] = old_key

            baseline = built.baseline
            self.assertIsInstance(baseline, GEPABaseline)
            assert isinstance(baseline, GEPABaseline)
            self.assertEqual(baseline.reflection_lm, "openai/glm-4-plus")
            self.assertEqual(baseline.reflection_lm_kwargs["api_base"], "https://api.z.ai/api/coding/paas/v4")
            self.assertEqual(baseline.reflection_lm_kwargs["api_key"], "test-glm-key")

    def test_gepa_update_reports_unchanged_when_native_best_candidate_matches_seed(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "gepa_project"
                package = project / "src" / "gepa"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("", encoding="utf-8")
                (package / "optimize_anything.py").write_text(
                    textwrap.dedent(
                        """
    class EngineConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ReflectionConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class GEPAConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeResult:
        best_candidate = "seed"
        num_candidates = 1
        best_idx = 0
        val_aggregate_scores = [1.0]

        def to_dict(self):
            return {"best_candidate": self.best_candidate, "num_candidates": self.num_candidates}

    def optimize_anything(**kwargs):
        return FakeResult()
                        """
                    ),
                    encoding="utf-8",
                )
                evaluator_module = root / "fake_gepa_evaluator.py"
                evaluator_module.write_text(
                    "def evaluate(**kwargs):\n    return 1.0\n",
                    encoding="utf-8",
                )
                sys.path.insert(0, str(root))
                for name in list(sys.modules):
                    if name == "gepa" or name.startswith("gepa."):
                        del sys.modules[name]
                try:
                    built = build_baseline(
                        {
                            "baseline": {
                                "name": "gepa",
                                "class_path": "seagym.baselines.gepa:GEPABaseline",
                                "config": {
                                    "project_dir": str(project),
                                    "seed_candidate": "seed",
                                    "evaluator_import_path": "fake_gepa_evaluator:evaluate",
                                },
                            }
                        },
                        run_dir=root / "run",
                        base_dir=Path.cwd(),
                    )
                    state = built.baseline.initialize(root / "run")
                    empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

                    result = built.baseline.update(empty_batch, state)

                    self.assertFalse(result.changed)
                    self.assertEqual(result.status, "unchanged")
                    self.assertEqual(result.metrics["num_candidates"], 1)
                    self.assertEqual(Path(result.artifacts["candidate_path"]).read_text(encoding="utf-8"), "seed")
                finally:
                    sys.path.remove(str(root))

    def test_gepa_update_captures_reflection_lm_token_usage(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "gepa_project"
                package = project / "src" / "gepa"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("", encoding="utf-8")
                (package / "optimize_anything.py").write_text(
                    textwrap.dedent(
                        """
    class EngineConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ReflectionConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class GEPAConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeLM:
        total_tokens_in = 12
        total_tokens_out = 8
        total_cost = 0.05

    class FakeResult:
        best_candidate = "seed improved"
        num_candidates = 2
        best_idx = 0
        val_aggregate_scores = [1.0]

        def to_dict(self):
            return {"best_candidate": self.best_candidate, "num_candidates": self.num_candidates}

    def optimize_anything(**kwargs):
        kwargs["config"].reflection.reflection_lm = FakeLM()
        return FakeResult()
                        """
                    ),
                    encoding="utf-8",
                )
                evaluator_module = root / "fake_gepa_evaluator.py"
                evaluator_module.write_text("def evaluate(**kwargs):\n    return 1.0\n", encoding="utf-8")
                sys.path.insert(0, str(root))
                for name in list(sys.modules):
                    if name == "gepa" or name.startswith("gepa."):
                        del sys.modules[name]
                try:
                    built = build_baseline(
                        {
                            "baseline": {
                                "name": "gepa",
                                "class_path": "seagym.baselines.gepa:GEPABaseline",
                                "config": {
                                    "project_dir": str(project),
                                    "seed_candidate": "seed",
                                    "evaluator_import_path": "fake_gepa_evaluator:evaluate",
                                },
                            }
                        },
                        run_dir=root / "run",
                        base_dir=Path.cwd(),
                    )
                    state = built.baseline.initialize(root / "run")
                    empty_batch = TrajectoryBatch(trajectories=[], task_ids=[], view_name="train", mode="train")

                    result = built.baseline.update(empty_batch, state)

                    self.assertEqual(
                        result.logs["cost"],
                        {"total_tokens": 20.0, "input_tokens": 12.0, "output_tokens": 8.0, "cost_usd": 0.05},
                    )
                    self.assertEqual(result.logs["cost_source"], "gepa_reflection_lm")
                finally:
                    sys.path.remove(str(root))

    def test_gepa_candidate_rendering_matches_native_string_unwrap(self) -> None:
            self.assertEqual(_render_candidate("seed"), "seed")
            self.assertEqual(_render_candidate({"prompt": "seed"}), '{\n  "prompt": "seed"\n}')
            self.assertEqual(_render_candidate({"instruction_prompt": "seed"}, component_key="instruction_prompt"), "seed")

    def test_gepa_terminal_bench_native_adapter_uses_official_adapter_shape(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "gepa_project"
                package = project / "src" / "gepa"
                adapter_package = package / "adapters" / "terminal_bench_adapter"
                example_dir = package / "examples" / "terminal-bench"
                adapter_package.mkdir(parents=True)
                (example_dir / "prompt-templates").mkdir(parents=True)
                (example_dir / "train_terminus.py").write_text("class TerminusWrapper: pass\n", encoding="utf-8")
                (example_dir / "prompt-templates" / "instruction_prompt.txt").write_text("seed", encoding="utf-8")
                (example_dir / "prompt-templates" / "terminus.txt").write_text("{{ instruction }}", encoding="utf-8")
                (package / "__init__.py").write_text(
                    textwrap.dedent(
                        """
    class EvaluationBatch:
        def __init__(self, outputs, scores, trajectories):
            self.outputs = outputs
            self.scores = scores
            self.trajectories = trajectories

    def optimize(**kwargs):
        import os
        trainset = kwargs["trainset"]
        adapter = kwargs["adapter"]
        assert trainset[0].task_id == "demo-task"
        assert trainset[0].model_name == "fake-model"
        assert adapter.n_concurrent == 2
        assert "terminal_bench_adapter/prompt-templates/instruction_prompt.txt" in adapter.instruction_prompt_path
        assert os.getcwd().endswith("terminal_bench_adapter")
        class FakeResult:
            best_candidate = {"instruction_prompt": "seed improved"}
            num_candidates = 2
            best_idx = 0
            val_aggregate_scores = [1.0]
            def to_dict(self):
                return {"best_candidate": self.best_candidate, "num_candidates": self.num_candidates}
        return FakeResult()
                        """
                    ),
                    encoding="utf-8",
                )
                (package / "adapters" / "__init__.py").write_text("", encoding="utf-8")
                (adapter_package / "__init__.py").write_text("", encoding="utf-8")
                (adapter_package / "terminal_bench_adapter.py").write_text(
                    textwrap.dedent(
                        """
    class TerminalBenchTask:
        def __init__(self, task_id, model_name):
            self.task_id = task_id
            self.model_name = model_name

    class TerminusAdapter:
        def __init__(self, n_concurrent=1, instruction_prompt_path="prompt.txt"):
            self.n_concurrent = n_concurrent
            self.instruction_prompt_path = instruction_prompt_path

    def get_results(task_id, run_id):
        return True, 1, "none", []

    def run_agent_tb(*args, **kwargs):
        return 0
                        """
                    ),
                    encoding="utf-8",
                )
                for name in list(sys.modules):
                    if name == "gepa" or name.startswith("gepa."):
                        del sys.modules[name]
                built = build_baseline(
                    {
                        "baseline": {
                            "name": "gepa",
                            "class_path": "seagym.baselines.gepa:GEPABaseline",
                            "config": {
                                "project_dir": str(project),
                                "seed_candidate": "seed",
                                "native_adapter": {
                                    "type": "terminal_bench",
                                    "model_name": "fake-model",
                                    "n_concurrent": 2,
                                },
                            },
                        }
                    },
                    run_dir=root / "run",
                    base_dir=Path.cwd(),
                )
                state = built.baseline.initialize(root / "run")
                batch = TrajectoryBatch(
                    trajectories=[
                        Trajectory(
                            task_id="terminal-bench/demo-task",
                            attempt_id=None,
                            view_name="train",
                            mode="train",
                            success=True,
                            reward=1.0,
                            score=1.0,
                            rewards={"reward": 1.0},
                        )
                    ],
                    task_ids=["terminal-bench/demo-task"],
                    view_name="train",
                    mode="train",
                )

                result = built.baseline.update(batch, state)

                self.assertTrue(result.changed)
                self.assertEqual(result.metrics["num_candidates"], 2)
                self.assertEqual(Path(result.artifacts["candidate_path"]).read_text(encoding="utf-8"), "seed improved")

    def test_gepa_harbor_native_adapter_evaluates_candidates_via_rollout_agent(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "gepa_project"
                package = project / "src" / "gepa"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(
                    textwrap.dedent(
                        """
    class EvaluationBatch:
        def __init__(self, outputs, scores, trajectories):
            self.outputs = outputs
            self.scores = scores
            self.trajectories = trajectories

    def optimize(**kwargs):
        adapter = kwargs["adapter"]
        batch = kwargs["trainset"]
        evaluation = adapter.evaluate(batch, kwargs["seed_candidate"], capture_traces=True)
        reflective = adapter.make_reflective_dataset(
            kwargs["seed_candidate"],
            evaluation,
            ["instruction_prompt"],
        )
        assert evaluation.scores == [1.0]
        assert reflective["instruction_prompt"][0]["Task ID"] == "terminal-bench/demo-task"
        class FakeResult:
            best_candidate = {"instruction_prompt": "harbor improved"}
            num_candidates = 2
            best_idx = 1
            val_aggregate_scores = [0.0, 1.0]
            def to_dict(self):
                return {"best_candidate": self.best_candidate, "num_candidates": self.num_candidates}
        return FakeResult()
                        """
                    ),
                    encoding="utf-8",
                )
                for name in list(sys.modules):
                    if name == "gepa" or name.startswith("gepa."):
                        del sys.modules[name]
                built = build_baseline(
                    {
                        "baseline": {
                            "name": "gepa",
                            "class_path": "seagym.baselines.gepa:GEPABaseline",
                            "config": {
                                "project_dir": str(project),
                                "seed_candidate": "seed",
                                "native_adapter": {
                                    "type": "harbor",
                                    "reflection_minibatch_size": 1,
                                },
                            },
                        }
                    },
                    run_dir=root / "run",
                    base_dir=Path.cwd(),
                )
                rollout_agent = _FakeGEPARolloutAgent()
                baseline = built.baseline
                assert isinstance(baseline, GEPABaseline)
                baseline.bind_runtime(env=object(), task_index=object(), rollout_agent=rollout_agent, run_dir=root / "run")
                state = baseline.initialize(root / "run")
                batch = TrajectoryBatch(
                    trajectories=[
                        Trajectory(
                            task_id="terminal-bench/demo-task",
                            attempt_id=None,
                            view_name="train",
                            mode="train",
                            success=False,
                            reward=0.0,
                            score=0.0,
                            rewards={"reward": 0.0},
                        )
                    ],
                    task_ids=["terminal-bench/demo-task"],
                    view_name="train",
                    mode="train",
                )

                result = baseline.update(batch, state)

                self.assertTrue(result.changed)
                self.assertEqual(Path(result.artifacts["candidate_path"]).read_text(encoding="utf-8"), "harbor improved")
                self.assertEqual(rollout_agent.task_ids, ["terminal-bench/demo-task"])
                self.assertEqual(rollout_agent.prompt_text, "seed")

    def test_gepa_harbor_native_adapter_uses_batch_plan_valset_view(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = root / "gepa_project"
                package = project / "src" / "gepa"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(
                    textwrap.dedent(
                        """
    class EvaluationBatch:
        def __init__(self, outputs, scores, trajectories):
            self.outputs = outputs
            self.scores = scores
            self.trajectories = trajectories

    def optimize(**kwargs):
        adapter = kwargs["adapter"]
        assert kwargs["trainset"] == ["train-a", "train-b"]
        assert kwargs["valset"] == ["val-a", "val-b"]
        assert hasattr(adapter, "propose_new_texts")
        assert adapter.propose_new_texts is None
        evaluation = adapter.evaluate(kwargs["trainset"], kwargs["seed_candidate"], capture_traces=True)
        reflective = adapter.make_reflective_dataset(
            kwargs["seed_candidate"],
            evaluation,
            ["instruction_prompt"],
        )
        assert len(evaluation.trajectories) == 2
        assert len(reflective["instruction_prompt"]) == 1
        class FakeResult:
            best_candidate = {"instruction_prompt": "harbor improved"}
            num_candidates = 2
            best_idx = 1
            val_aggregate_scores = [0.0, 1.0]
            def to_dict(self):
                return {"best_candidate": self.best_candidate, "num_candidates": self.num_candidates}
        return FakeResult()
                        """
                    ),
                    encoding="utf-8",
                )
                for name in list(sys.modules):
                    if name == "gepa" or name.startswith("gepa."):
                        del sys.modules[name]
                built = build_baseline(
                    {
                        "baseline": {
                            "name": "gepa",
                            "class_path": "seagym.baselines.gepa:GEPABaseline",
                            "config": {
                                "project_dir": str(project),
                                "seed_candidate": "seed",
                                "native_adapter": {
                                    "type": "harbor",
                                    "max_reflective_records": 1,
                                    "valset_view": "update_validation",
                                    "reflection_minibatch_size": 20,
                                },
                            },
                        }
                    },
                    run_dir=root / "run",
                    base_dir=Path.cwd(),
                )

                class FakeBatchPlan:
                    views = {"update_validation": ["val-a", "val-b"]}

                baseline = built.baseline
                assert isinstance(baseline, GEPABaseline)
                baseline.bind_runtime(
                    env=object(),
                    task_index=object(),
                    rollout_agent=_FakeGEPARolloutAgent(),
                    run_dir=root / "run",
                    batch_plan=FakeBatchPlan(),
                )
                state = baseline.initialize(root / "run")
                batch = TrajectoryBatch(
                    trajectories=[
                        Trajectory(
                            task_id="train-a",
                            attempt_id=None,
                            view_name="train",
                            mode="train",
                            success=False,
                            reward=0.0,
                            score=0.0,
                            rewards={"reward": 0.0},
                        ),
                        Trajectory(
                            task_id="train-b",
                            attempt_id=None,
                            view_name="train",
                            mode="train",
                            success=False,
                            reward=0.0,
                            score=0.0,
                            rewards={"reward": 0.0},
                        ),
                    ],
                    task_ids=["train-a", "train-b"],
                    view_name="train",
                    mode="train",
                )

                result = baseline.update(batch, state)

                self.assertTrue(result.changed)

class _FakeGEPARolloutAgent:
    agent_id = "fake-harbor-agent"

    def __init__(self) -> None:
        self.task_ids: list[str] = []
        self.prompt_text = ""

    def rollout(self, batch, *, env, task_index, baseline_state):
        del env, task_index
        self.task_ids = list(batch.task_ids)
        self.prompt_text = Path(baseline_state.metadata["prompt_template_path"]).read_text(encoding="utf-8")
        return TrajectoryBatch.from_task_results(
            [
                TaskRunResult(
                    task_id=task_id,
                    view_name=batch.view_name,
                    mode=batch.mode,
                    rewards={"reward": 1.0},
                    score=1.0,
                    success=True,
                    refs={"trial_name": f"{task_id}__trial"},
                )
                for task_id in batch.task_ids
            ],
            task_ids=batch.task_ids,
            view_name=batch.view_name,
            mode=batch.mode,
        )

if __name__ == "__main__":
    unittest.main()
