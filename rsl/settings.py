import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_set(name: str) -> frozenset[str]:
    raw = os.environ.get(name, "")
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class RSLSettings:
    """Runtime knobs for the online RSL decision stage.

    Every value has a conservative default: no exploration, candidates may cost
    at most 1.5x the baseline estimate, and no model allowlist/denylist.
    """

    epsilon: float = 0.0
    max_cost_ratio: float = 1.5
    max_estimated_cost: float | None = None
    allowlist: frozenset[str] = field(default_factory=frozenset)
    denylist: frozenset[str] = field(default_factory=frozenset)
    canary_salt: str = "rsl-canary"
    state_ttl_seconds: float = 5.0
    default_completion_tokens: int = 512

    @classmethod
    def from_env(cls) -> "RSLSettings":
        epsilon = min(1.0, max(0.0, _env_float("ROUTER_RSL_EPSILON", 0.0)))
        ttl = max(0.0, _env_float("ROUTER_RSL_STATE_TTL_SECONDS", 5.0))
        return cls(
            epsilon=epsilon,
            max_cost_ratio=max(0.0, _env_float("ROUTER_RSL_MAX_COST_RATIO", 1.5)),
            max_estimated_cost=_env_optional_float("ROUTER_RSL_MAX_ESTIMATED_COST"),
            allowlist=_env_set("ROUTER_RSL_ALLOWLIST"),
            denylist=_env_set("ROUTER_RSL_DENYLIST"),
            canary_salt=os.environ.get("ROUTER_RSL_CANARY_SALT", "rsl-canary") or "rsl-canary",
            state_ttl_seconds=ttl,
            default_completion_tokens=max(
                1, int(_env_float("ROUTER_RSL_DEFAULT_COMPLETION_TOKENS", 512))
            ),
        )
