import logging
import uuid
from typing import Any

from app.core.request_context import RequestContext

logger = logging.getLogger("model_router.rsl")


def _message_text(messages: list[Any]) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def extract_features(messages: list[Any]) -> dict[str, Any]:
    text = _message_text(messages)
    length = len(text)
    if length < 1000:
        length_bucket = "short"
    elif length < 8000:
        length_bucket = "medium"
    else:
        length_bucket = "long"
    return {
        "message_count": len(messages),
        "input_characters": length,
        "input_length_bucket": length_bucket,
        "contains_code_fence": "```" in text,
        "contains_json_hint": "json" in text.lower(),
    }


def classify_task(recommender, messages: list[Any], request_id: str | None = None) -> str:
    if recommender is None or not hasattr(recommender, "profile_task"):
        return "unknown"
    try:
        return recommender.profile_task(messages).name or "unknown"
    except Exception:
        logger.exception(
            "Unable to classify RSL request %s; using unknown task bucket",
            request_id,
        )
        return "unknown"


def ensure_request_id(request_context: RequestContext) -> str:
    if not request_context.request_id:
        request_context.request_id = str(uuid.uuid4())
    return request_context.request_id


def build_baseline_decision(
    request_context: RequestContext,
    body,
    *,
    task_type: str,
    requested_model: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": ensure_request_id(request_context),
        "api_key_id": request_context.api_key_id,
        "subject_type": request_context.subject_type,
        "subject_id": request_context.subject_id,
        "user_id": request_context.user_id,
        "routing_profile_id": request_context.routing_profile_id,
        "requested_model": requested_model or body.model,
        "baseline_model": body.model,
        "selected_model": body.model,
        "proposed_model": None,
        "task_type": task_type,
        "features": extract_features(body.messages),
        "mode": "baseline",
        "behavior_policy": "baseline_only",
        "policy_version": None,
        "propensity": 1.0,
        "applied": False,
        "eligible": True,
        "skip_reason": None,
    }


def build_outcome(
    *,
    request_id: str,
    response,
    body,
    model_config,
    status: str,
    latency_ms: int,
    error: Exception | None = None,
    fallback_reason: str | None = None,
    executed_model: str | None = None,
    actual_cost_override: float | None = None,
    override_actual_cost: bool = False,
) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    cascade = getattr(response, "cascade", None)
    pricing = getattr(model_config, "pricing", None)
    actual_cost = actual_cost_override if override_actual_cost else None
    if not override_actual_cost and usage is not None and pricing is not None:
        actual_cost = (
            (usage.prompt_tokens / 1_000_000) * pricing.prompt
            + (usage.completion_tokens / 1_000_000) * pricing.completion
        )

    return {
        "request_id": request_id,
        "executed_model": executed_model or getattr(response, "actual_model", None) or body.model,
        "status": status,
        "error_type": type(error).__name__ if error is not None else None,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "actual_cost": actual_cost,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "empty_response": content is not None and not content.strip(),
        "judge_score": getattr(cascade, "judge_score", None),
        "quality_score": None,
        "quality_source": None,
        "fallback_reason": fallback_reason,
    }


async def record_outcome_safely(store, outcome: dict[str, Any]) -> None:
    try:
        await store.record_outcome(outcome)
    except Exception:
        logger.exception(
            "Unable to persist RSL outcome for request %s",
            outcome.get("request_id"),
        )
