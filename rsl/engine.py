"""Online RSL decision entry point used by ``app/api/chat.py``.

The engine runs after the existing router has produced a baseline model and
before the existing access check. It records every decision and, depending on
the rollout stage, either only proposes a model (shadow) or replaces the
baseline (canary/active). It never widens the set of models the request may use.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from rsl.collector import build_baseline_decision, classify_task, ensure_request_id
from rsl.guardrails import (
    CandidateSet,
    build_candidate_set,
    enforce_selection,
    estimate_prompt_tokens,
)
from rsl.rollout import RolloutState, resolve_mode
from rsl.selector import select_action
from rsl.settings import RSLSettings

logger = logging.getLogger("model_router.rsl")


@dataclass(slots=True)
class RSLDecision:
    request_id: str
    baseline_model: str
    selected_model: str
    mode: str = "baseline"
    applied: bool = False
    proposed_model: str | None = None
    policy_version: str | None = None
    propensity: float = 1.0
    behavior_policy: str = "baseline_only"
    skip_reason: str | None = None
    rollout_bucket: int | None = None
    task_type: str = "unknown"
    candidate_set: CandidateSet | None = None
    fallback_reason: str | None = None
    recorded: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def executes_policy(self) -> bool:
        return self.behavior_policy != "baseline_only"

    def response_metadata(self, executed_model: str | None) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "policy_version": self.policy_version,
            "mode": self.mode,
            "baseline_model": self.baseline_model,
            "selected_model": self.selected_model,
            "executed_model": executed_model,
            "propensity": self.propensity,
            "applied": self.applied,
            "fallback_reason": self.fallback_reason,
            "fallback_from_model": self.details.get("fallback_from_model"),
        }


class RSLEngine:
    def __init__(
        self,
        *,
        store,
        router_engine,
        settings: RSLSettings | None = None,
        recommender=None,
        clock=time.monotonic,
        rng: random.Random | None = None,
    ):
        self.store = store
        self.router_engine = router_engine
        self.settings = settings or RSLSettings()
        self.recommender = recommender
        self._clock = clock
        self._rng = rng
        self._state: RolloutState | None = None
        self._state_loaded_at: float | None = None
        self._policy_cache: dict[str, dict[str, Any] | None] = {}

    async def rollout_state(self, *, force: bool = False) -> RolloutState:
        now = self._clock()
        stale = (
            self._state is None
            or self._state_loaded_at is None
            or now - self._state_loaded_at > self.settings.state_ttl_seconds
        )
        if force or stale:
            row = await self.store.get_rollout_state()
            self._state = RolloutState.from_row(row)
            self._state_loaded_at = now
            self._policy_cache.clear()
        return self._state

    def invalidate(self) -> None:
        self._state = None
        self._state_loaded_at = None
        self._policy_cache.clear()

    async def load_policy(self, policy_version: str) -> dict[str, Any] | None:
        if policy_version in self._policy_cache:
            return self._policy_cache[policy_version]
        row = await self.store.get_policy(policy_version)
        policy = row.get("policy") if row else None
        self._policy_cache[policy_version] = policy
        return policy

    async def decide(
        self,
        *,
        request_context,
        body,
        routing_policy,
        requested_model: str,
        replace_allowed: bool,
        candidate_models: set[str] | None = None,
        recommender=None,
    ) -> RSLDecision:
        request_id = ensure_request_id(request_context)
        task_type = classify_task(recommender or self.recommender, body.messages, request_id)
        record = build_baseline_decision(
            request_context,
            body,
            task_type=task_type,
            requested_model=requested_model,
        )
        decision = RSLDecision(
            request_id=request_id,
            baseline_model=body.model,
            selected_model=body.model,
            task_type=task_type,
        )

        state = await self.rollout_state()
        if state.stage != "disabled" and state.policy_version:
            if not replace_allowed:
                decision.skip_reason = "request_not_managed"
            else:
                await self._apply_policy(
                    decision,
                    state=state,
                    request_context=request_context,
                    body=body,
                    routing_policy=routing_policy,
                    candidate_models=candidate_models,
                )
        else:
            decision.skip_reason = "rollout_disabled"

        record.update(
            {
                "selected_model": decision.selected_model,
                "proposed_model": decision.proposed_model,
                "mode": decision.mode,
                "behavior_policy": decision.behavior_policy,
                "policy_version": decision.policy_version,
                "propensity": decision.propensity,
                "applied": decision.applied,
                "skip_reason": decision.skip_reason,
                "rollout_bucket": decision.rollout_bucket,
                "action_distribution": (
                    decision.details.get("selection", {}).get("action_distribution", {})
                    if decision.behavior_policy != "baseline_only"
                    else {decision.selected_model: 1.0}
                ),
            }
        )
        persistence_error = False
        try:
            decision.recorded = await self.store.record_decision(record)
        except Exception:
            persistence_error = True
            logger.exception("Unable to persist RSL decision for request %s", request_id)
        if not decision.recorded:
            if not persistence_error:
                logger.warning(
                    "RSL decision for request %s already exists; retaining baseline model",
                    request_id,
                )
            decision.selected_model = decision.baseline_model
            decision.mode = "baseline"
            decision.applied = False
            decision.proposed_model = None
            decision.policy_version = None
            decision.propensity = 1.0
            decision.behavior_policy = "baseline_only"
            decision.skip_reason = (
                "decision_persistence_failed"
                if persistence_error
                else "request_id_conflict"
            )
            decision.rollout_bucket = None
            decision.candidate_set = None
            decision.details = {}
        return decision

    async def _apply_policy(
        self,
        decision: RSLDecision,
        *,
        state: RolloutState,
        request_context,
        body,
        routing_policy,
        candidate_models,
    ) -> None:
        policy = await self.load_policy(state.policy_version)
        decision.policy_version = state.policy_version
        if policy is None:
            decision.skip_reason = "policy_missing"
            return

        subject_key = f"{request_context.subject_type}:{request_context.subject_id}"
        resolution = resolve_mode(state, subject_key, self.settings.canary_salt)
        decision.mode = resolution.mode
        decision.rollout_bucket = resolution.bucket

        candidate_set = self.preview_candidates(
            routing_policy=routing_policy,
            request_allowed_models=request_context.allowed_models,
            baseline_model=decision.baseline_model,
            messages=body.messages,
            max_tokens=getattr(body, "max_tokens", None),
        )
        if candidate_models is not None:
            candidate_set.rejected.update(
                {
                    model: "not_in_request_mode_action_set"
                    for model in candidate_set.candidates
                    if model not in candidate_models
                }
            )
            candidate_set.candidates = [
                model for model in candidate_set.candidates if model in candidate_models
            ]
        decision.candidate_set = candidate_set
        selection = select_action(
            policy,
            decision.task_type,
            candidate_set,
            epsilon=self.settings.epsilon if resolution.executes_policy else 0.0,
            rng=self._rng,
        )
        proposed, rejection = enforce_selection(
            selection.model,
            candidate_set,
            routing_policy=routing_policy,
            request_allowed_models=request_context.allowed_models,
        )
        decision.proposed_model = proposed
        decision.details = {
            "selection": selection.to_dict(),
            "guardrail_rejection": rejection,
            "rejected_candidates": candidate_set.rejected,
        }

        if not resolution.executes_policy:
            decision.skip_reason = resolution.skip_reason
            return
        if rejection:
            decision.skip_reason = rejection
            return

        decision.selected_model = proposed
        decision.applied = proposed != decision.baseline_model
        decision.propensity = selection.propensity
        decision.behavior_policy = f"rsl:{state.policy_version}"
        if not decision.applied:
            decision.skip_reason = selection.reason

    def preview_candidates(
        self,
        *,
        routing_policy,
        request_allowed_models: set[str] | None,
        baseline_model: str,
        messages: list[Any],
        max_tokens: int | None = None,
    ) -> CandidateSet:
        return build_candidate_set(
            router_engine=self.router_engine,
            routing_policy=routing_policy,
            request_allowed_models=request_allowed_models,
            baseline_model=baseline_model,
            settings=self.settings,
            prompt_tokens=estimate_prompt_tokens(messages),
            completion_tokens=int(max_tokens or self.settings.default_completion_tokens),
        )
