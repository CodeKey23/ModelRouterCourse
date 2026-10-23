from typing import Any


POSITIVE_LABELS = {"good", "positive", "success", "accepted"}
NEGATIVE_LABELS = {"bad", "negative", "failure", "rejected"}


def _unit_interval(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def score_sample(
    sample: dict[str, Any],
    *,
    quality_weight: float = 1.0,
    cost_weight: float = 0.3,
    latency_weight: float = 0.1,
    failure_penalty: float = 1.0,
    cost_scale: float = 0.05,
    latency_scale_ms: float = 5000.0,
) -> dict[str, Any]:
    rating = sample.get("feedback_rating")
    label = str(sample.get("feedback_label") or "").strip().lower()

    if rating is not None:
        quality = _unit_interval(float(rating))
        quality_source = sample.get("feedback_source") or "feedback_rating"
    elif label in POSITIVE_LABELS:
        quality = 1.0
        quality_source = "feedback_label"
    elif label in NEGATIVE_LABELS:
        quality = 0.0
        quality_source = "feedback_label"
    elif sample.get("judge_score") is not None:
        quality = _unit_interval(float(sample["judge_score"]))
        quality_source = "cascade_judge"
    elif sample.get("status") != "success":
        quality = 0.0
        quality_source = "provider_failure"
    elif sample.get("empty_response"):
        quality = 0.0
        quality_source = "empty_response"
    elif sample.get("finish_reason") == "length":
        quality = 0.25
        quality_source = "truncated_heuristic"
    else:
        quality = 0.5
        quality_source = "neutral_heuristic"

    cost = sample.get("actual_cost")
    normalized_cost = (
        min(1.0, max(0.0, float(cost) / cost_scale))
        if cost is not None and cost_scale > 0
        else None
    )
    latency = sample.get("latency_ms")
    normalized_latency = (
        min(1.0, max(0.0, float(latency) / latency_scale_ms))
        if latency is not None and latency_scale_ms > 0
        else None
    )
    failed = sample.get("status") != "success"

    reward = quality_weight * quality
    if normalized_cost is not None:
        reward -= cost_weight * normalized_cost
    if normalized_latency is not None:
        reward -= latency_weight * normalized_latency
    if failed:
        reward -= failure_penalty

    return {
        **sample,
        "quality": quality,
        "quality_source": quality_source,
        "normalized_cost": normalized_cost,
        "normalized_latency": normalized_latency,
        "failed": failed,
        "reward": reward,
    }
