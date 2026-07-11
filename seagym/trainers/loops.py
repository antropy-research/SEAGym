from __future__ import annotations

"""Standard SEAGym update-validation training loop."""

from dataclasses import asdict, dataclass

from seagym.envs import TaskRunResult
from seagym.trainers.engine import EvaluationPoint, ExecutionEngine
from seagym.trainers.checkpoint import TrainerState


@dataclass
class UpdateValidationLoop:
    """Run the standard train-batch + frozen update-validation protocol."""

    def run(self, engine: ExecutionEngine, *, resume: TrainerState | None = None) -> None:
        val_view = engine.materialize_view("update_validation")
        assert isinstance(val_view, list)
        train_batches = engine.make_train_batches()
        updates_per_batch = engine.context.config.schedule.num_updates_per_batch
        if updates_per_batch <= 0:
            raise ValueError("schedule.num_updates_per_batch must be positive")
        total_updates = len(train_batches) * updates_per_batch
        batches_per_epoch = _batches_per_epoch(
            num_batches=len(train_batches),
            num_epochs=engine.context.config.schedule.num_epochs,
        )
        print(
            "seagym progress: loop initialized "
            f"train_batches={len(train_batches)} "
            f"updates_per_batch={updates_per_batch} "
            f"derived_num_updates={total_updates} "
            f"batches_per_epoch={batches_per_epoch} "
            f"update_validation_tasks={len(val_view)}",
            flush=True,
        )
        if resume is None:
            e0 = engine.record_evaluation_point(
                point_type="initial",
                train_batch_index=0,
                num_train_tasks_seen=0,
            )
            initial_checkpoint = engine.save_checkpoint(
                "initial",
                checkpoint_type="initial",
                trainer_state=TrainerState(
                    epoch=0,
                    train_batch_index=0,
                    global_step=0,
                    updates_completed=0,
                    num_train_tasks_seen=0,
                    checkpoint_id="initial",
                ),
                metadata={"kind": "initial", "epoch_index": 0, "train_batch_index": 0, "num_train_tasks_seen": 0},
            )
            previous_val_results = []
            if val_view:
                previous_val_results = engine.run_tasks(
                    val_view,
                    view_name="update_validation",
                    mode="validation",
                    evaluation_point=e0,
                )
            initial_evaluations = {}
            if val_view:
                initial_evaluations["update_validation"] = summarize_view(
                    "V_update-val", val_view, previous_val_results
                )
            engine.write_evaluation_point(
                e0,
                evaluations=initial_evaluations,
                refs={
                    "metric_inputs": str(engine.metric_input_path),
                    "agent_checkpoint": initial_checkpoint,
                },
            )
            seen = 0
            updates_completed = 0
            start_after_batch = 0
        else:
            if resume.train_batch_index < 0 or resume.train_batch_index > len(train_batches):
                raise ValueError("resume.train_batch_index must be between 0 and the train batch count")
            if resume.updates_completed < 0 or resume.updates_completed > total_updates:
                raise ValueError("resume.updates_completed must be between 0 and derived_num_updates")
            if not resume.checkpoint_id:
                raise ValueError("resume.checkpoint_id is required")
            load_ref = engine.load_checkpoint(resume.checkpoint_id)
            previous_val_results = [_task_result_from_dict(item) for item in (resume.previous_update_validation_results or [])]
            seen = resume.num_train_tasks_seen
            updates_completed = resume.updates_completed
            start_after_batch = resume.train_batch_index
            print(
                "seagym progress: loop resumed "
                f"checkpoint={resume.checkpoint_id} batch={start_after_batch}/{len(train_batches)} "
                f"seen={seen} updates_completed={updates_completed}/{total_updates} "
                f"load_ref={load_ref.get('loaded')}",
                flush=True,
            )

        last_epoch_checkpoint_id: str | None = None
        for step, batch in enumerate(train_batches, start=1):
            if step <= start_after_batch:
                continue
            update_ref: dict[str, object] | None = None
            for repeat_index in range(1, updates_per_batch + 1):
                print(
                    "seagym progress: train batch rollout started "
                    f"batch={step}/{len(train_batches)} repeat={repeat_index}/{updates_per_batch} "
                    f"tasks={len(batch)} seen_before={seen}",
                    flush=True,
                )
                seen_after_rollout = seen + len(batch)
                train_results = engine.run_tasks(
                    batch,
                    view_name="train",
                    mode="train",
                    train_batch_index=step,
                    update_repeat_index=repeat_index,
                    num_updates_per_batch=updates_per_batch,
                    global_update_index=updates_completed + 1,
                    num_train_tasks_seen=seen_after_rollout,
                )
                train_successes = sum(1 for result in train_results if result.success)
                train_score = (
                    0.0 if not train_results else sum(result.score for result in train_results) / len(train_results)
                )
                seen = seen_after_rollout
                print(
                    "seagym progress: train batch rollout finished "
                    f"batch={step}/{len(train_batches)} repeat={repeat_index}/{updates_per_batch} "
                    f"tasks={len(batch)} successes={train_successes} mean_score={train_score:.6f} seen={seen}",
                    flush=True,
                )
                print(
                    "seagym progress: baseline update started "
                    f"batch={step}/{len(train_batches)} repeat={repeat_index}/{updates_per_batch} "
                    f"update={updates_completed + 1}/{total_updates}",
                    flush=True,
                )
                update_ref = engine.update_agent(
                    train_results,
                    train_batch_index=step,
                    num_train_tasks_seen=seen,
                    update_repeat_index=repeat_index,
                    num_updates_per_batch=updates_per_batch,
                    global_update_index=updates_completed + 1,
                )
                updates_completed += 1
                update_summary = update_ref.get("summary") if isinstance(update_ref, dict) else {}
                update_status = update_summary.get("status", "unknown") if isinstance(update_summary, dict) else "unknown"
                update_changed = (
                    update_summary.get("changed", "unknown") if isinstance(update_summary, dict) else "unknown"
                )
                print(
                    "seagym progress: baseline update finished "
                    f"batch={step}/{len(train_batches)} repeat={repeat_index}/{updates_per_batch} "
                    f"update={updates_completed}/{total_updates} status={update_status} changed={update_changed}",
                    flush=True,
                )
            assert update_ref is not None

            step_checkpoint_id: str | None = None
            is_epoch_boundary = step % batches_per_epoch == 0
            if is_epoch_boundary:
                epoch_index = step // batches_per_epoch
                checkpoint_id = f"epoch_{epoch_index:04d}"
                checkpoint_state = TrainerState(
                    epoch=epoch_index,
                    train_batch_index=step,
                    global_step=updates_completed,
                    updates_completed=updates_completed,
                    num_train_tasks_seen=seen,
                    checkpoint_id=checkpoint_id,
                    previous_update_validation_results=[asdict(result) for result in previous_val_results],
                )
                engine.save_checkpoint(
                    checkpoint_id,
                    checkpoint_type="epoch",
                    trainer_state=checkpoint_state,
                    metadata={
                        "kind": "epoch",
                        "epoch_index": epoch_index,
                        "train_batch_index": step,
                        "num_train_tasks_seen": seen,
                    },
                )
                last_epoch_checkpoint_id = checkpoint_id
                step_checkpoint_id = checkpoint_id
                print(
                    "seagym progress: epoch checkpoint saved "
                    f"epoch={epoch_index}/{engine.context.config.schedule.num_epochs} "
                    f"checkpoint={checkpoint_id} batch={step}/{len(train_batches)}",
                    flush=True,
                )

            if val_view and is_epoch_boundary:
                print(
                    "seagym progress: update validation started "
                    f"batch={step}/{len(train_batches)} epoch={step // batches_per_epoch} tasks={len(val_view)}",
                    flush=True,
                )
                point: EvaluationPoint | None = engine.record_evaluation_point(
                    point_type="update_validation",
                    train_batch_index=step,
                    num_train_tasks_seen=seen,
                )
                checkpoint_metadata = {
                    "kind": "evaluation_point",
                    "point_type": point.point_type,
                    "train_batch_index": step,
                    "num_train_tasks_seen": seen,
                }
                if step_checkpoint_id is None:
                    checkpoint_ref = engine.save_checkpoint(
                        point.evaluation_point_id,
                        checkpoint_type="evaluation_point",
                        trainer_state=TrainerState(
                            epoch=step // batches_per_epoch,
                            train_batch_index=step,
                            global_step=updates_completed,
                            updates_completed=updates_completed,
                            num_train_tasks_seen=seen,
                            checkpoint_id=point.evaluation_point_id,
                            previous_update_validation_results=[asdict(result) for result in previous_val_results],
                        ),
                        metadata=checkpoint_metadata,
                    )
                else:
                    checkpoint_ref = engine.alias_checkpoint(
                        point.evaluation_point_id,
                        source_checkpoint_id=step_checkpoint_id,
                        checkpoint_type="evaluation_point",
                        trainer_state=TrainerState(
                            epoch=step // batches_per_epoch,
                            train_batch_index=step,
                            global_step=updates_completed,
                            updates_completed=updates_completed,
                            num_train_tasks_seen=seen,
                            checkpoint_id=point.evaluation_point_id,
                            previous_update_validation_results=[asdict(result) for result in previous_val_results],
                        ),
                        metadata={**checkpoint_metadata, "alias_of": step_checkpoint_id},
                    )
                current = engine.run_tasks(
                    val_view,
                    view_name="update_validation",
                    mode="validation",
                    evaluation_point=point,
                )
                update_assessment = engine.assess_update(current, previous_val_results)
                previous_val_results = current
                print(
                    "seagym progress: update validation finished "
                    f"batch={step}/{len(train_batches)} assessment={update_assessment.get('label', 'unknown')}",
                    flush=True,
                )
            else:
                point = None
                update_assessment = {"label": "not_applicable"}
                current = []

            replay_point: EvaluationPoint | None = None
            if engine.should_replay(step):
                replay_view = engine.materialize_view("replay")
                assert isinstance(replay_view, list)
                if point is None:
                    replay_point = engine.record_evaluation_point(
                        point_type="replay",
                        train_batch_index=step,
                        num_train_tasks_seen=seen,
                    )
                    replay_checkpoint_metadata = {
                        "kind": "evaluation_point",
                        "point_type": replay_point.point_type,
                        "train_batch_index": step,
                        "num_train_tasks_seen": seen,
                    }
                    if step_checkpoint_id is None:
                        replay_checkpoint_ref = engine.save_checkpoint(
                            replay_point.evaluation_point_id,
                            checkpoint_type="evaluation_point",
                            trainer_state=TrainerState(
                                epoch=step // batches_per_epoch,
                                train_batch_index=step,
                                global_step=updates_completed,
                                updates_completed=updates_completed,
                                num_train_tasks_seen=seen,
                                checkpoint_id=replay_point.evaluation_point_id,
                                previous_update_validation_results=[asdict(result) for result in previous_val_results],
                            ),
                            metadata=replay_checkpoint_metadata,
                        )
                    else:
                        replay_checkpoint_ref = engine.alias_checkpoint(
                            replay_point.evaluation_point_id,
                            source_checkpoint_id=step_checkpoint_id,
                            checkpoint_type="evaluation_point",
                            trainer_state=TrainerState(
                                epoch=step // batches_per_epoch,
                                train_batch_index=step,
                                global_step=updates_completed,
                                updates_completed=updates_completed,
                                num_train_tasks_seen=seen,
                                checkpoint_id=replay_point.evaluation_point_id,
                                previous_update_validation_results=[asdict(result) for result in previous_val_results],
                            ),
                            metadata={**replay_checkpoint_metadata, "alias_of": step_checkpoint_id},
                        )
                    replay_results = engine.run_tasks(
                        replay_view,
                        view_name="replay",
                        mode="replay",
                        evaluation_point=replay_point,
                        agent_checkpoint_id=replay_point.evaluation_point_id,
                    )
                else:
                    replay_results = engine.run_tasks(
                        replay_view,
                        view_name="replay",
                        mode="replay",
                        evaluation_point=point,
                        agent_checkpoint_id=point.evaluation_point_id,
                    )
            else:
                replay_results = []

            if point is not None:
                evaluations = {
                    "update_validation": summarize_view("V_update-val", val_view, current),
                }
                if replay_results:
                    replay_view = engine.materialize_view("replay")
                    assert isinstance(replay_view, list)
                    evaluations["replay"] = summarize_view("replay", replay_view, replay_results)
                engine.write_evaluation_point(
                    point,
                    evaluations=evaluations,
                    refs={
                        "metric_inputs": str(engine.metric_input_path),
                        "agent_update": update_ref,
                        "agent_checkpoint": checkpoint_ref,
                    },
                    update_assessment=update_assessment,
                )
            elif replay_results:
                replay_view = engine.materialize_view("replay")
                assert isinstance(replay_view, list)
                assert replay_point is not None
                engine.write_evaluation_point(
                    replay_point,
                    evaluations={"replay": summarize_view("replay", replay_view, replay_results)},
                    refs={
                        "metric_inputs": str(engine.metric_input_path),
                        "agent_update": update_ref,
                        "agent_checkpoint": replay_checkpoint_ref,
                    },
                )

        final_ep = engine.record_evaluation_point(
            point_type="final",
            train_batch_index=len(train_batches),
            num_train_tasks_seen=seen,
        )
        print(
            "seagym progress: final evaluation started "
            f"views={len(engine.final_views())}",
            flush=True,
        )
        final_metadata = {
            "kind": "final",
            "epoch_index": engine.context.config.schedule.num_epochs,
            "train_batch_index": len(train_batches),
            "num_train_tasks_seen": seen,
        }
        if last_epoch_checkpoint_id is not None:
            final_checkpoint = engine.alias_checkpoint(
                "final",
                source_checkpoint_id=last_epoch_checkpoint_id,
                checkpoint_type="final",
                trainer_state=TrainerState(
                    epoch=engine.context.config.schedule.num_epochs,
                    train_batch_index=len(train_batches),
                    global_step=updates_completed,
                    updates_completed=updates_completed,
                    num_train_tasks_seen=seen,
                    checkpoint_id="final",
                    previous_update_validation_results=[asdict(result) for result in previous_val_results],
                ),
                metadata={**final_metadata, "alias_of": last_epoch_checkpoint_id},
            )
        else:
            final_checkpoint = engine.save_checkpoint(
                "final",
                checkpoint_type="final",
                trainer_state=TrainerState(
                    epoch=engine.context.config.schedule.num_epochs,
                    train_batch_index=len(train_batches),
                    global_step=updates_completed,
                    updates_completed=updates_completed,
                    num_train_tasks_seen=seen,
                    checkpoint_id="final",
                    previous_update_validation_results=[asdict(result) for result in previous_val_results],
                ),
                metadata=final_metadata,
            )
        final_evaluations: dict[str, dict[str, object]] = {}
        for view_name, task_ids in engine.final_views().items():
            at_results = engine.run_tasks(
                task_ids,
                view_name=view_name,
                mode="final",
                evaluation_point=final_ep,
                agent_checkpoint_id="final",
                baseline_role="A_T",
            )
            a0_load = engine.load_checkpoint("initial")
            a0_results = engine.run_tasks(
                task_ids,
                view_name=view_name,
                mode="final_baseline",
                evaluation_point=final_ep,
                agent_checkpoint_id="initial",
                baseline_role="A_0",
            )
            at_reload = engine.load_checkpoint("final")
            final_evaluations[view_name] = summarize_final_view(view_name, task_ids, at_results, a0_results)
            final_evaluations[view_name]["refs"] = {
                "A_0_load": a0_load,
                "A_T_reload": at_reload,
            }
        engine.write_evaluation_point(
            final_ep,
            evaluations=final_evaluations,
            refs={
                "metric_inputs": str(engine.metric_input_path),
                "agent_checkpoint": final_checkpoint,
            },
        )
        print("seagym progress: final evaluation finished", flush=True)


def summarize_view(view_ref: str, task_ids: list[str], results: list[TaskRunResult]) -> dict[str, object]:
    score = 0.0 if not results else sum(result.score for result in results) / len(results)
    return {
        "view_ref": view_ref,
        "subset_id": "+".join(task_ids),
        "score": score,
        "num_tasks": len(task_ids),
    }


def summarize_final_view(
    view_ref: str,
    task_ids: list[str],
    at_results: list[TaskRunResult],
    a0_results: list[TaskRunResult],
) -> dict[str, object]:
    summary = summarize_view(view_ref, task_ids, at_results)
    baseline_score = 0.0 if not a0_results else sum(result.score for result in a0_results) / len(a0_results)
    summary.update(
        {
            "agent_checkpoint_id": "A_T",
            "baseline_checkpoint_id": "A_0",
            "baseline_score": baseline_score,
            "gain_vs_A_0": float(summary["score"]) - baseline_score,
            "num_baseline_tasks": len(a0_results),
        }
    )
    return summary


def _task_result_from_dict(data: dict[str, object]) -> TaskRunResult:
    rewards = data.get("rewards") if isinstance(data.get("rewards"), dict) else {}
    cost = data.get("cost") if isinstance(data.get("cost"), dict) else {}
    refs = data.get("refs") if isinstance(data.get("refs"), dict) else {}
    return TaskRunResult(
        task_id=str(data.get("task_id", "")),
        view_name=str(data.get("view_name", "")),
        mode=str(data.get("mode", "")),
        rewards={str(key): float(value) for key, value in rewards.items()},
        score=float(data.get("score", 0.0)),
        success=bool(data.get("success", False)),
        cost={str(key): float(value) for key, value in cost.items()},
        error=None if data.get("error") in (None, "") else str(data.get("error")),
        refs=refs,
    )


def _batches_per_epoch(*, num_batches: int, num_epochs: int) -> int:
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if num_batches % num_epochs != 0:
        raise ValueError(
            "Cannot infer epoch boundaries: materialized train batch count "
            f"{num_batches} is not divisible by num_epochs {num_epochs}"
        )
    batches_per_epoch = num_batches // num_epochs
    if batches_per_epoch <= 0:
        raise ValueError("Cannot infer epoch boundaries from an empty train batch stream")
    return batches_per_epoch
