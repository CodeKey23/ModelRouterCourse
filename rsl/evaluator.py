import json
import math
from collections import defaultdict
from typing import Any

_NORMAL_95 = 1.959963984540054


def _action_distribution(sample: dict[str, Any]) -> dict[str, float] | None:
    raw = sample.get("action_distribution")
    if raw is None:
        raw = sample.get("action_distribution_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not raw and sample.get("behavior_policy") == "baseline_only":
        baseline = sample.get("baseline_model")
        if baseline:
            return {str(baseline): 1.0}
    if not isinstance(raw, dict) or not raw:
        return None

    distribution: dict[str, float] = {}
    for action, probability in raw.items():
        try:
            value = float(probability)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return None
        distribution[str(action)] = value
    if not math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        return None
    return distribution


def _weighted_estimate(
    samples: list[dict[str, Any]],
    target_actions: list[str],
) -> tuple[float | None, float, float, list[float]]:
    weighted_rewards: list[float] = []
    weights: list[float] = []
    influence_values: list[float] = []
    for sample, target_action in zip(samples, target_actions):
        distribution = _action_distribution(sample)
        if distribution is None:
            return None, 0.0, 0.0, []
        probability = distribution.get(target_action, 0.0)
        if probability <= 0.0:
            return None, 0.0, 0.0, []

        logged_action = str(sample.get("selected_model") or sample.get("baseline_model") or "")
        logged_probability = distribution.get(logged_action, 0.0)
        try:
            propensity = float(sample.get("propensity") or 0.0)
            reward = float(sample.get("reward", 0.0))
        except (TypeError, ValueError):
            return None, 0.0, 0.0, []
        if (
            not math.isfinite(reward)
            or not math.isfinite(propensity)
            or propensity <= 0.0
            or propensity > 1.0
            or not math.isclose(propensity, logged_probability, rel_tol=1e-7, abs_tol=1e-9)
        ):
            return None, 0.0, 0.0, []

        weight = 1.0 / probability if logged_action == target_action else 0.0
        weights.append(weight)
        weighted_rewards.append(weight * reward)

    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return None, 0.0, 0.0, []
    estimate = sum(weighted_rewards) / weight_sum
    squared_weight_sum = sum(weight * weight for weight in weights)
    effective_sample_size = (
        weight_sum * weight_sum / squared_weight_sum if squared_weight_sum > 0.0 else 0.0
    )
    influence_values = [
        (weighted_reward - estimate * weight) * len(samples) / weight_sum
        for weighted_reward, weight in zip(weighted_rewards, weights)
    ]
    return estimate, effective_sample_size, weight_sum, influence_values


def _cluster_standard_error(
    influence_values: list[float],
    samples: list[dict[str, Any]],
) -> float | None:
    clusters: dict[str, float] = defaultdict(float)
    for index, (influence, sample) in enumerate(zip(influence_values, samples)):
        subject_id = sample.get("subject_id")
        cluster = (
            f"{sample.get('subject_type') or 'subject'}:{subject_id}"
            if subject_id
            else str(sample.get("request_id") or f"sample:{index}")
        )
        clusters[cluster] += influence
    cluster_count = len(clusters)
    if cluster_count < 2:
        return None
    variance = (
        cluster_count
        / (cluster_count - 1)
        * sum(value * value for value in clusters.values())
        / (len(samples) * len(samples))
    )
    return math.sqrt(variance)


def _cluster_count(samples: list[dict[str, Any]]) -> int:
    return len(
        {
            (
                f"{sample.get('subject_type') or 'subject'}:{sample['subject_id']}"
                if sample.get("subject_id")
                else str(sample.get("request_id") or f"sample:{index}")
            )
            for index, sample in enumerate(samples)
        }
    )


def evaluate_policy(
    samples: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    min_effective_samples: float = 30.0,
    min_independent_clusters: int = 30,
) -> dict[str, Any]:
    """Evaluate a deterministic policy with held-out, logged action probabilities.

    SNIPS estimates are reported only when both the candidate and baseline
    actions have positive logged probability for every sample in each task.
    """
    task_actions = policy.get("task_actions", {})
    task_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_baseline_only = True
    for sample in samples:
        task = str(sample.get("task_type") or "unknown")
        task_samples[task].append(sample)
        if sample.get("behavior_policy") != "baseline_only":
            all_baseline_only = False

    report: dict[str, dict[str, Any]] = {}
    task_deltas: list[tuple[str, int, float, list[float]]] = []
    total_supported = 0
    reasons: set[str] = set()
    for task, rows in sorted(task_samples.items()):
        recommendation = (task_actions.get(task) or {}).get("recommended_model", "baseline")
        policy_actions = [
            str(row.get("baseline_model") or "")
            if recommendation == "baseline"
            else str(recommendation)
            for row in rows
        ]
        baseline_actions = [str(row.get("baseline_model") or "") for row in rows]

        policy_estimate, policy_ess, _, policy_influence = _weighted_estimate(
            rows, policy_actions
        )
        baseline_estimate, baseline_ess, _, baseline_influence = _weighted_estimate(
            rows, baseline_actions
        )
        policy_supported = policy_estimate is not None
        baseline_supported = baseline_estimate is not None
        support_ratio = 1.0 if policy_supported and baseline_supported else 0.0
        effective_samples = min(policy_ess, baseline_ess)
        independent_clusters = _cluster_count(rows)
        task_result: dict[str, Any] = {
            "samples": len(rows),
            "supported_samples": len(rows) if support_ratio else 0,
            "support_ratio": support_ratio,
            "policy_action": recommendation,
            "policy_estimate": policy_estimate,
            "baseline_estimate": baseline_estimate,
            "effective_sample_size": effective_samples,
            "independent_clusters": independent_clusters,
        }

        if not policy_supported or not baseline_supported:
            reason = (
                "baseline_only_logging_has_no_counterfactual_action_support"
                if all_baseline_only and recommendation != "baseline"
                else "candidate_policy_lacks_action_support"
            )
            task_result["reason"] = reason
            reasons.add(reason)
        elif effective_samples < min_effective_samples:
            task_result["reason"] = "insufficient_effective_sample_size"
            reasons.add("insufficient_effective_sample_size")
        elif independent_clusters < min_independent_clusters:
            task_result["reason"] = "insufficient_independent_clusters"
            reasons.add("insufficient_independent_clusters")
        else:
            deltas = [
                policy_value - baseline_value
                for policy_value, baseline_value in zip(policy_influence, baseline_influence)
            ]
            delta = policy_estimate - baseline_estimate
            standard_error = _cluster_standard_error(deltas, rows)
            if standard_error is None:
                task_result["reason"] = "insufficient_independent_clusters"
                reasons.add("insufficient_independent_clusters")
                report[task] = task_result
                continue
            task_result.update(
                {
                    "uplift": delta,
                    "standard_error": standard_error,
                    "confidence_interval_95": [
                        delta - _NORMAL_95 * standard_error,
                        delta + _NORMAL_95 * standard_error,
                    ],
                }
            )
            task_deltas.append((task, len(rows), delta, deltas))
            total_supported += len(rows)
        report[task] = task_result

    sample_count = sum(len(rows) for rows in task_samples.values())
    if sample_count == 0:
        reasons.add("no_evaluation_samples")
    causal_available = sample_count > 0 and len(task_deltas) == len(task_samples)

    result: dict[str, Any] = {
        "status": "evaluated" if causal_available else "not_evaluable",
        "causal_estimate_available": causal_available,
        "reason": None if causal_available else sorted(reasons)[0],
        "sample_count": sample_count,
        "supported_sample_count": total_supported,
        "support_ratio": total_supported / sample_count if sample_count else 0.0,
        "min_effective_samples": min_effective_samples,
        "min_independent_clusters": min_independent_clusters,
        "confidence_level": 0.95,
        "tasks": report,
    }
    if causal_available:
        all_samples = [sample for rows in task_samples.values() for sample in rows]
        if _cluster_count(all_samples) < min_independent_clusters:
            result.update(
                {
                    "status": "not_evaluable",
                    "causal_estimate_available": False,
                    "reason": "insufficient_independent_clusters",
                }
            )
            return result
        policy_estimate = sum(
            len(rows) * float(report[task]["policy_estimate"])
            for task, rows in task_samples.items()
        ) / sample_count
        baseline_estimate = sum(
            len(rows) * float(report[task]["baseline_estimate"])
            for task, rows in task_samples.items()
        ) / sample_count
        uplift = sum(count * delta for _, count, delta, _ in task_deltas) / sample_count
        overall_influences: list[float] = []
        overall_samples: list[dict[str, Any]] = []
        for task, _, _, influences in task_deltas:
            overall_influences.extend(influences)
            overall_samples.extend(task_samples[task])
        standard_error = _cluster_standard_error(overall_influences, overall_samples)
        if standard_error is None:
            result.update(
                {
                    "status": "not_evaluable",
                    "causal_estimate_available": False,
                    "reason": "insufficient_independent_clusters",
                }
            )
            return result
        result.update(
            {
                "policy_estimate": policy_estimate,
                "baseline_estimate": baseline_estimate,
                "uplift": uplift,
                "standard_error": standard_error,
                "confidence_interval_95": [
                    uplift - _NORMAL_95 * standard_error,
                    uplift + _NORMAL_95 * standard_error,
                ],
            }
        )
    return result
