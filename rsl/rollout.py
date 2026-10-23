"""Rollout stages, canary bucketing and transition rules.

Stages: ``disabled → shadow → canary → active``; ``rollback`` returns to the
previous active policy or, when none exists, to ``disabled``.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

STAGES = ("disabled", "shadow", "canary", "active")
BUCKET_COUNT = 10_000


@dataclass(slots=True)
class RolloutState:
    stage: str = "disabled"
    policy_version: str | None = None
    canary_percent: float = 0.0
    previous_policy_version: str | None = None
    updated_at: str | None = None
    note: str | None = None
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "policy_version": self.policy_version,
            "canary_percent": self.canary_percent,
            "previous_policy_version": self.previous_policy_version,
            "updated_at": self.updated_at,
            "note": self.note,
            "revision": self.revision,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "RolloutState":
        if not row:
            return cls()
        return cls(
            stage=row.get("stage") or "disabled",
            policy_version=row.get("policy_version"),
            canary_percent=float(row.get("canary_percent") or 0.0),
            previous_policy_version=row.get("previous_policy_version"),
            updated_at=row.get("updated_at"),
            note=row.get("note"),
            revision=int(row.get("revision") or 0),
        )


@dataclass(slots=True)
class ModeResolution:
    mode: str
    executes_policy: bool
    skip_reason: str | None = None
    bucket: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


def assign_bucket(subject_key: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{subject_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % BUCKET_COUNT


def in_canary(bucket: int, percent: float) -> bool:
    threshold = int(round(max(0.0, min(100.0, percent)) * BUCKET_COUNT / 100.0))
    return bucket < threshold


def resolve_mode(state: RolloutState, subject_key: str, salt: str) -> ModeResolution:
    if state.stage == "disabled" or not state.policy_version:
        return ModeResolution(mode="baseline", executes_policy=False, skip_reason="rollout_disabled")
    if state.stage == "shadow":
        return ModeResolution(mode="shadow", executes_policy=False)
    if state.stage == "canary":
        bucket = assign_bucket(subject_key, salt)
        if in_canary(bucket, state.canary_percent):
            return ModeResolution(mode="canary", executes_policy=True, bucket=bucket)
        return ModeResolution(
            mode="canary",
            executes_policy=False,
            skip_reason="outside_canary_bucket",
            bucket=bucket,
        )
    if state.stage == "active":
        return ModeResolution(mode="active", executes_policy=True)
    return ModeResolution(mode="baseline", executes_policy=False, skip_reason="unknown_stage")


class RolloutTransitionError(ValueError):
    pass


def _policy_proven(policy_row: dict[str, Any]) -> bool:
    evaluation = policy_row.get("evaluation") or {}
    return bool(evaluation.get("causal_estimate_available"))


def plan_transition(
    current: RolloutState,
    *,
    target_stage: str,
    policy_row: dict[str, Any] | None,
    canary_percent: float | None = None,
    acknowledge_unproven: bool = False,
) -> RolloutState:
    """Validate an operator request and return the next rollout state."""
    if target_stage not in STAGES:
        raise RolloutTransitionError(f"Unknown rollout stage '{target_stage}'")

    if target_stage == "disabled":
        return RolloutState(
            stage="disabled",
            policy_version=None,
            canary_percent=0.0,
            previous_policy_version=_last_active(current),
        )

    if policy_row is None:
        raise RolloutTransitionError("Policy version was not found")
    version = str(policy_row["policy_version"])
    status = str(policy_row.get("status") or "candidate")
    if status in {"rolled_back", "retired"}:
        raise RolloutTransitionError(f"Policy '{version}' is {status} and cannot be promoted")

    if target_stage == "shadow":
        if current.stage == "active" and current.policy_version != version:
            raise RolloutTransitionError(
                "Another policy is active; disable or roll it back before shadowing a new one"
            )
        return RolloutState(
            stage="shadow",
            policy_version=version,
            canary_percent=0.0,
            previous_policy_version=_last_active(current),
        )

    if not acknowledge_unproven and not _policy_proven(policy_row):
        raise RolloutTransitionError(
            "Policy has no causal evaluation; set acknowledge_unproven=true to expose it to traffic"
        )

    if target_stage == "canary":
        if canary_percent is None or not (0.0 < canary_percent < 100.0):
            raise RolloutTransitionError("canary_percent must be between 0 and 100 (exclusive)")
        same_policy = current.policy_version == version
        if not (same_policy and current.stage in {"shadow", "canary"}):
            raise RolloutTransitionError(
                f"Policy '{version}' must be in shadow before canary"
            )
        return RolloutState(
            stage="canary",
            policy_version=version,
            canary_percent=float(canary_percent),
            previous_policy_version=_last_active(current),
        )

    # target_stage == "active"
    if not (current.policy_version == version and current.stage == "canary"):
        raise RolloutTransitionError(
            f"Policy '{version}' must complete canary before activation"
        )
    return RolloutState(
        stage="active",
        policy_version=version,
        canary_percent=100.0,
        previous_policy_version=_last_active(current),
    )


def plan_rollback(
    current: RolloutState,
    *,
    target_policy_row: dict[str, Any] | None,
) -> RolloutState:
    """Roll back to a previously active policy or to disabled."""
    if target_policy_row is None:
        return RolloutState(
            stage="disabled",
            policy_version=None,
            canary_percent=0.0,
            previous_policy_version=None,
        )
    version = str(target_policy_row["policy_version"])
    if version == current.policy_version and current.stage == "active":
        raise RolloutTransitionError("Target policy is already active")
    if target_policy_row.get("status") not in {"active", "superseded"}:
        raise RolloutTransitionError(
            f"Policy '{version}' was never active and cannot be a rollback target"
        )
    return RolloutState(
        stage="active",
        policy_version=version,
        canary_percent=100.0,
        previous_policy_version=None,
    )


def _last_active(current: RolloutState) -> str | None:
    if current.stage == "active" and current.policy_version:
        return current.policy_version
    return current.previous_policy_version
