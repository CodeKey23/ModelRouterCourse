import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import ceil
from statistics import mean
from typing import Any

from rsl.evaluator import evaluate_policy
from rsl.scoring import score_sample


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


async def train(
    store,
    *,
    window_days: int = 7,
    feedback_wait_hours: int = 24,
    min_samples_per_cell: int = 30,
    evaluation_fraction: float = 0.25,
    min_effective_samples: float = 30.0,
    min_independent_clusters: int = 30,
    now: datetime | None = None,
) -> dict[str, Any]:
    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    if feedback_wait_hours < 0:
        raise ValueError("feedback_wait_hours must not be negative")
    if min_samples_per_cell < 1:
        raise ValueError("min_samples_per_cell must be at least 1")
    if not 0.0 < evaluation_fraction < 0.5:
        raise ValueError("evaluation_fraction must be between 0 and 0.5")
    if min_effective_samples < 1:
        raise ValueError("min_effective_samples must be at least 1")
    if min_independent_clusters < 2:
        raise ValueError("min_independent_clusters must be at least 2")

    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    start = end - timedelta(days=window_days)
    outcomes_before = end - timedelta(hours=feedback_wait_hours)
    window_start = _iso_utc(start)
    window_end = _iso_utc(end)
    rows = await store.get_training_rows(
        window_start=window_start,
        window_end=window_end,
        outcomes_before=_iso_utc(outcomes_before),
    )
    if not rows:
        raise ValueError("No completed RSL baseline samples in the selected window")

    scored_samples = [score_sample(row) for row in rows]
    if len(scored_samples) > 1:
        evaluation_count = max(1, ceil(len(scored_samples) * evaluation_fraction))
        split_at = len(scored_samples) - evaluation_count
        training_samples = scored_samples[:split_at]
        evaluation_samples = scored_samples[split_at:]
    else:
        training_samples = scored_samples
        evaluation_samples = []
    frozen_samples = [
        {**sample, "data_partition": "training"}
        for sample in training_samples
    ] + [
        {**sample, "data_partition": "evaluation"}
        for sample in evaluation_samples
    ]
    dataset_id = f"dataset-{uuid.uuid4().hex}"
    await store.create_dataset(
        dataset_id=dataset_id,
        window_start=window_start,
        window_end=window_end,
        filters={
            "eligible_only": True,
            "behavior_policies": sorted(
                {str(row.get("behavior_policy") or "baseline_only") for row in rows}
            ),
            "feedback_wait_hours": feedback_wait_hours,
            "scoring_version": 1,
            "evaluation_fraction": evaluation_fraction,
            "min_effective_samples": min_effective_samples,
            "min_independent_clusters": min_independent_clusters,
            "evaluation_strategy": "chronological_holdout",
        },
        samples=frozen_samples,
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in training_samples:
        task = sample.get("task_type") or "unknown"
        # The logged action is what the behavior policy chose to execute.
        action = sample.get("selected_model") or sample.get("baseline_model") or "unknown"
        grouped[(task, action)].append(sample)
    behavior_policies = {
        str(sample.get("behavior_policy") or "baseline_only") for sample in training_samples
    }
    training_mode = (
        "baseline_only" if behavior_policies == {"baseline_only"} else "mixed_logging"
    )

    task_groups: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (task, action), action_samples in grouped.items():
        task_groups[task].append((action, action_samples))

    task_actions: dict[str, dict[str, Any]] = {}
    for task, action_groups in sorted(task_groups.items()):
        ranked = sorted(
            action_groups,
            key=lambda item: (
                -mean(float(sample["reward"]) for sample in item[1]),
                item[0],
            ),
        )
        winner, winner_samples = ranked[0]
        recommended_model = (
            winner
            if len(winner_samples) >= min_samples_per_cell
            else "baseline"
        )
        task_actions[task] = {
            "recommended_model": recommended_model,
            "observed_actions": [
                {
                    "model": action,
                    "samples": len(action_samples),
                    "mean_reward": mean(float(item["reward"]) for item in action_samples),
                    "mean_quality": mean(float(item["quality"]) for item in action_samples),
                    "failure_rate": mean(float(item["failed"]) for item in action_samples),
                }
                for action, action_samples in ranked
            ],
            "deployable": False,
            "deployment_note": (
                "Observational rewards are descriptive; exposure to traffic "
                "requires explicit operator acknowledgement via the rollout API."
            ),
        }

    parent_version = await store.latest_policy_version()
    policy_version = f"rsl-candidate-{uuid.uuid4().hex[:12]}"
    policy = {
        "policy_version": policy_version,
        "parent_version": parent_version,
        "dataset_id": dataset_id,
        "training_mode": training_mode,
        "scoring_version": 1,
        "task_actions": task_actions,
        "deployable": False,
    }
    evaluation = evaluate_policy(
        evaluation_samples,
        policy,
        min_effective_samples=min_effective_samples,
        min_independent_clusters=min_independent_clusters,
    )
    metrics = {
        "sample_count": len(scored_samples),
        "training_sample_count": len(training_samples),
        "evaluation_sample_count": len(evaluation_samples),
        "task_count": len(task_actions),
        "action_count": len(grouped),
        "mean_reward": (
            mean(float(sample["reward"]) for sample in training_samples)
            if training_samples
            else None
        ),
        "quality_sources": _count_sources(scored_samples),
        "evaluation_status": evaluation["status"],
    }
    await store.save_policy(
        policy_version=policy_version,
        parent_version=parent_version,
        dataset_id=dataset_id,
        policy=policy,
        metrics=metrics,
        evaluation=evaluation,
    )
    return {
        "policy_version": policy_version,
        "dataset_id": dataset_id,
        "metrics": metrics,
        "evaluation": evaluation,
        "policy": policy,
    }


def _count_sources(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        source = str(sample.get("quality_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))
