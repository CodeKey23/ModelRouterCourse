"""Hard constraints that bound every RSL action.

The guardrails never grant access: they only shrink the candidate set. The
original routing policy / request access check in ``app/api/chat.py`` still
runs on the final model after RSL has made its choice.
"""

from dataclasses import dataclass, field
from typing import Any

from rsl.settings import RSLSettings


@dataclass(slots=True)
class CandidateSet:
    baseline_model: str
    candidates: list[str]
    rejected: dict[str, str] = field(default_factory=dict)
    baseline_estimated_cost: float | None = None
    estimated_costs: dict[str, float | None] = field(default_factory=dict)

    @property
    def actions(self) -> list[str]:
        """Action space: baseline first, then the surviving candidates."""
        ordered = [self.baseline_model]
        ordered.extend(model for model in self.candidates if model != self.baseline_model)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "candidates": list(self.candidates),
            "rejected": dict(self.rejected),
            "baseline_estimated_cost": self.baseline_estimated_cost,
            "estimated_costs": dict(self.estimated_costs),
        }


def estimate_prompt_tokens(messages: list[Any]) -> int:
    characters = 0
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            characters += len(content)
    # Rough pre-execution estimate; actual usage comes from the provider.
    return max(1, characters // 4)


def estimate_cost(model_config, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = getattr(model_config, "pricing", None)
    if pricing is None:
        return None
    prompt_price = getattr(pricing, "prompt", None)
    completion_price = getattr(pricing, "completion", None)
    if prompt_price is None or completion_price is None:
        return None
    return (prompt_tokens / 1_000_000) * float(prompt_price) + (
        completion_tokens / 1_000_000
    ) * float(completion_price)


def _model_available(router_engine, routing_key: str) -> bool:
    checker = getattr(router_engine, "is_model_available", None)
    if checker is None:
        return True
    try:
        return bool(checker(routing_key))
    except Exception:
        return False


def build_candidate_set(
    *,
    router_engine,
    routing_policy,
    request_allowed_models: set[str] | None,
    baseline_model: str,
    settings: RSLSettings,
    prompt_tokens: int,
    completion_tokens: int,
) -> CandidateSet:
    model_map = getattr(router_engine, "model_map", {}) or {}
    baseline_entry = model_map.get(baseline_model)
    baseline_cost = (
        estimate_cost(baseline_entry[1], prompt_tokens, completion_tokens)
        if baseline_entry
        else None
    )
    result = CandidateSet(
        baseline_model=baseline_model,
        candidates=[],
        baseline_estimated_cost=baseline_cost,
    )

    for routing_key in sorted(model_map.keys()):
        if routing_key == baseline_model:
            continue
        reason = _reject_reason(
            routing_key=routing_key,
            model_config=model_map[routing_key][1],
            router_engine=router_engine,
            routing_policy=routing_policy,
            request_allowed_models=request_allowed_models,
            settings=settings,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            baseline_cost=baseline_cost,
            result=result,
        )
        if reason:
            result.rejected[routing_key] = reason
        else:
            result.candidates.append(routing_key)
    return result


def _reject_reason(
    *,
    routing_key: str,
    model_config,
    router_engine,
    routing_policy,
    request_allowed_models: set[str] | None,
    settings: RSLSettings,
    prompt_tokens: int,
    completion_tokens: int,
    baseline_cost: float | None,
    result: CandidateSet,
) -> str | None:
    if not routing_policy.is_model_allowed(routing_key, request_allowed_models):
        return "not_allowed_by_policy_or_request"
    if settings.allowlist and routing_key not in settings.allowlist:
        return "not_in_rsl_allowlist"
    if routing_key in settings.denylist:
        return "in_rsl_denylist"
    if not _model_available(router_engine, routing_key):
        return "model_unavailable"

    cost = estimate_cost(model_config, prompt_tokens, completion_tokens)
    result.estimated_costs[routing_key] = cost
    if settings.max_estimated_cost is not None:
        if cost is None:
            return "pricing_unknown_with_cost_budget"
        if cost > settings.max_estimated_cost:
            return "exceeds_estimated_cost_budget"
    if (
        baseline_cost is not None
        and baseline_cost > 0
        and cost is not None
        and settings.max_cost_ratio > 0
        and cost > baseline_cost * settings.max_cost_ratio
    ):
        return "exceeds_baseline_cost_ratio"
    return None


def enforce_selection(
    selected_model: str,
    candidate_set: CandidateSet,
    *,
    routing_policy,
    request_allowed_models: set[str] | None,
) -> tuple[str, str | None]:
    """Return the model that may actually run and a rejection reason if any."""
    if selected_model == candidate_set.baseline_model:
        return selected_model, None
    if selected_model not in candidate_set.candidates:
        return candidate_set.baseline_model, "selected_not_in_candidates"
    if not routing_policy.is_model_allowed(selected_model, request_allowed_models):
        return candidate_set.baseline_model, "selected_not_allowed"
    return selected_model, None
