"""Policy lookup and action selection with exact propensities."""

import random
from dataclasses import dataclass
from typing import Any

from rsl.guardrails import CandidateSet


@dataclass(slots=True)
class Selection:
    model: str
    propensity: float
    greedy_model: str
    reason: str
    explored: bool = False
    action_distribution: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "propensity": self.propensity,
            "greedy_model": self.greedy_model,
            "reason": self.reason,
            "explored": self.explored,
            "action_distribution": self.action_distribution or {},
        }


def greedy_action(policy: dict[str, Any], task_type: str, candidate_set: CandidateSet) -> tuple[str, str]:
    task_actions = (policy or {}).get("task_actions") or {}
    entry = task_actions.get(task_type) or task_actions.get("unknown")
    if not entry:
        return candidate_set.baseline_model, "no_policy_entry_for_task"
    recommended = entry.get("recommended_model") or "baseline"
    if recommended == "baseline" or recommended == candidate_set.baseline_model:
        return candidate_set.baseline_model, "policy_recommends_baseline"
    if recommended not in candidate_set.candidates:
        return candidate_set.baseline_model, "recommended_model_rejected_by_guardrails"
    return recommended, "policy_recommendation"


def action_distribution(
    greedy: str,
    actions: list[str],
    epsilon: float,
) -> dict[str, float]:
    """Exact sampling distribution: (1-eps) on greedy plus eps spread uniformly."""
    if not actions:
        raise ValueError("actions must not be empty")
    epsilon = min(1.0, max(0.0, float(epsilon)))
    if len(actions) == 1:
        return {actions[0]: 1.0}
    uniform = epsilon / len(actions)
    distribution = {action: uniform for action in actions}
    distribution[greedy] = distribution.get(greedy, 0.0) + (1.0 - epsilon)
    return distribution


def select_action(
    policy: dict[str, Any],
    task_type: str,
    candidate_set: CandidateSet,
    *,
    epsilon: float = 0.0,
    rng: random.Random | None = None,
) -> Selection:
    greedy, reason = greedy_action(policy, task_type, candidate_set)
    actions = candidate_set.actions
    distribution = action_distribution(greedy, actions, epsilon)
    if epsilon <= 0.0 or len(actions) == 1:
        return Selection(
            model=greedy,
            propensity=distribution[greedy],
            greedy_model=greedy,
            reason=reason,
            action_distribution=distribution,
        )

    sampler = rng or random.SystemRandom()
    draw = sampler.random()
    cumulative = 0.0
    chosen = actions[-1]
    for action in actions:
        cumulative += distribution[action]
        if draw < cumulative:
            chosen = action
            break
    return Selection(
        model=chosen,
        propensity=distribution[chosen],
        greedy_model=greedy,
        reason=reason if chosen == greedy else "epsilon_exploration",
        explored=chosen != greedy,
        action_distribution=distribution,
    )
