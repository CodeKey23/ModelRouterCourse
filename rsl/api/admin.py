"""Admin-only RSL operations: inspection, preview, rollout and rollback.

Mounted under the existing ``/admin`` router, so the original admin bearer
token check (``require_admin``) applies to every endpoint here.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api.admin._auth import require_admin
from app.core.routing_policy import load_routing_policy
from rsl.guardrails import enforce_selection
from rsl.rollout import (
    RolloutState,
    RolloutTransitionError,
    plan_rollback,
    plan_transition,
)
from rsl.selector import select_action
from rsl.store import RolloutRevisionConflict

router = APIRouter(prefix="/rsl", tags=["admin / rsl"])


class RolloutRequest(BaseModel):
    policy_version: str = Field(min_length=1, max_length=128)
    note: str | None = Field(default=None, max_length=500)
    actor: str | None = Field(default=None, max_length=128)


class CanaryRequest(RolloutRequest):
    percent: float = Field(gt=0.0, lt=100.0)
    acknowledge_unproven: bool = False


class ActivateRequest(RolloutRequest):
    acknowledge_unproven: bool = False


class RollbackRequest(BaseModel):
    to_policy_version: str | None = Field(default=None, max_length=128)
    note: str | None = Field(default=None, max_length=500)
    actor: str | None = Field(default=None, max_length=128)


class DisableRequest(BaseModel):
    note: str | None = Field(default=None, max_length=500)
    actor: str | None = Field(default=None, max_length=128)


class PreviewMessage(BaseModel):
    role: str = "user"
    content: str


class PreviewRequest(BaseModel):
    policy_version: str | None = None
    task_type: str = "unknown"
    baseline_model: str = Field(min_length=1)
    routing_profile_id: str | None = None
    allowed_models: list[str] | None = None
    messages: list[PreviewMessage] = Field(default_factory=list)
    max_tokens: int | None = None


def _engine(request: Request):
    engine = getattr(request.app.state, "rsl_engine", None)
    if engine is None or not getattr(request.app.state, "rsl_recording_enabled", False):
        raise HTTPException(status_code=404, detail="RSL is disabled")
    return engine


async def _current_state(engine) -> RolloutState:
    return await engine.rollout_state(force=True)


async def _commit_transition(
    engine,
    *,
    current: RolloutState,
    new_state: RolloutState,
    action: str,
    actor: str | None,
    note: str | None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = engine.store
    policy_status_updates: list[tuple[str, str]] = []
    if current.policy_version and current.policy_version != new_state.policy_version:
        if action == "rollback":
            previous_status = "rolled_back"
        elif current.stage == "active":
            previous_status = "superseded"
        else:
            previous_status = "candidate"
        policy_status_updates.append((current.policy_version, previous_status))
    if new_state.policy_version:
        policy_status_updates.append((new_state.policy_version, new_state.stage))
    try:
        saved = await store.commit_rollout_transition(
            expected_revision=current.revision,
            stage=new_state.stage,
            policy_version=new_state.policy_version,
            canary_percent=new_state.canary_percent,
            previous_policy_version=new_state.previous_policy_version,
            note=note,
            policy_status_updates=policy_status_updates,
            action=action,
            from_stage=current.stage,
            from_policy_version=current.policy_version,
            actor=actor,
            details={
                "note": note,
                "from_policy_version": current.policy_version,
                "to_policy_version": new_state.policy_version,
                **(details or {}),
            },
        )
    except RolloutRevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    engine.invalidate()
    return {"ok": True, "rollout": saved}


@router.get("/status")
async def rsl_status(request: Request, _=Depends(require_admin)):
    engine = _engine(request)
    state = await _current_state(engine)
    summary = await engine.store.decision_summary(policy_version=state.policy_version)
    return {
        "rollout": state.to_dict(),
        "settings": {
            "epsilon": engine.settings.epsilon,
            "max_cost_ratio": engine.settings.max_cost_ratio,
            "max_estimated_cost": engine.settings.max_estimated_cost,
            "allowlist": sorted(engine.settings.allowlist),
            "denylist": sorted(engine.settings.denylist),
        },
        "decision_summary": summary,
    }


@router.get("/policies")
async def list_policies(request: Request, limit: int = Query(20, ge=1, le=200), _=Depends(require_admin)):
    engine = _engine(request)
    return await engine.store.list_policies(limit=limit)


@router.get("/policies/{policy_version}")
async def get_policy(policy_version: str, request: Request, _=Depends(require_admin)):
    engine = _engine(request)
    row = await engine.store.get_policy(policy_version)
    if row is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return row


@router.post("/preview")
async def preview_decision(body: PreviewRequest, request: Request, _=Depends(require_admin)):
    engine = _engine(request)
    state = await _current_state(engine)
    version = body.policy_version or state.policy_version
    if not version:
        raise HTTPException(status_code=400, detail="No policy version to preview")
    policy_row = await engine.store.get_policy(version)
    if policy_row is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    if body.baseline_model not in getattr(engine.router_engine, "model_map", {}):
        raise HTTPException(status_code=400, detail="baseline_model is not a loaded routing key")

    routing_policy = await load_routing_policy(request.app.state.db, body.routing_profile_id)
    allowed = set(body.allowed_models) if body.allowed_models is not None else None
    candidate_set = engine.preview_candidates(
        routing_policy=routing_policy,
        request_allowed_models=allowed,
        baseline_model=body.baseline_model,
        messages=body.messages,
        max_tokens=body.max_tokens,
    )
    selection = select_action(policy_row["policy"], body.task_type, candidate_set, epsilon=0.0)
    final_model, rejection = enforce_selection(
        selection.model,
        candidate_set,
        routing_policy=routing_policy,
        request_allowed_models=allowed,
    )
    return {
        "policy_version": version,
        "task_type": body.task_type,
        "candidate_set": candidate_set.to_dict(),
        "selection": selection.to_dict(),
        "final_model": final_model,
        "guardrail_rejection": rejection,
        "would_replace_baseline": final_model != body.baseline_model,
    }


async def _transition(request: Request, body, *, target_stage: str, action: str, **kwargs):
    engine = _engine(request)
    current = await _current_state(engine)
    policy_row = await engine.store.get_policy(body.policy_version)
    try:
        new_state = plan_transition(
            current,
            target_stage=target_stage,
            policy_row=policy_row,
            **kwargs,
        )
    except RolloutTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _commit_transition(
        engine,
        current=current,
        new_state=new_state,
        action=action,
        actor=body.actor,
        note=body.note,
        details={
            "canary_percent": new_state.canary_percent,
            "acknowledge_unproven": bool(getattr(body, "acknowledge_unproven", False)),
        },
    )


@router.post("/rollout/shadow")
async def rollout_shadow(body: RolloutRequest, request: Request, _=Depends(require_admin)):
    return await _transition(request, body, target_stage="shadow", action="shadow")


@router.post("/rollout/canary")
async def rollout_canary(body: CanaryRequest, request: Request, _=Depends(require_admin)):
    return await _transition(
        request,
        body,
        target_stage="canary",
        action="canary",
        canary_percent=body.percent,
        acknowledge_unproven=body.acknowledge_unproven,
    )


@router.post("/rollout/activate")
async def rollout_activate(body: ActivateRequest, request: Request, _=Depends(require_admin)):
    return await _transition(
        request,
        body,
        target_stage="active",
        action="activate",
        acknowledge_unproven=body.acknowledge_unproven,
    )


@router.post("/rollout/rollback")
async def rollout_rollback(body: RollbackRequest, request: Request, _=Depends(require_admin)):
    engine = _engine(request)
    current = await _current_state(engine)
    target_version = body.to_policy_version or current.previous_policy_version
    target_row = await engine.store.get_policy(target_version) if target_version else None
    if body.to_policy_version and target_row is None:
        raise HTTPException(status_code=404, detail="Rollback target policy not found")
    try:
        new_state = plan_rollback(current, target_policy_row=target_row)
    except RolloutTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _commit_transition(
        engine,
        current=current,
        new_state=new_state,
        action="rollback",
        actor=body.actor,
        note=body.note,
        details={"requested_target": body.to_policy_version},
    )


@router.post("/rollout/disable")
async def rollout_disable(body: DisableRequest, request: Request, _=Depends(require_admin)):
    engine = _engine(request)
    current = await _current_state(engine)
    new_state = plan_transition(current, target_stage="disabled", policy_row=None)
    return await _commit_transition(
        engine,
        current=current,
        new_state=new_state,
        action="disable",
        actor=body.actor,
        note=body.note,
    )


@router.get("/audit")
async def rsl_audit(request: Request, limit: int = Query(50, ge=1, le=500), _=Depends(require_admin)):
    engine = _engine(request)
    return await engine.store.list_audit(limit=limit)
