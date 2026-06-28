"""Immutable state primitives for the agentic loop.

Determinism principle: every phase consumes and produces explicit, serializable
state. Nothing is hidden in closures or globals, so any iteration can be logged,
diffed, and replayed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass(frozen=True)
class Observation:
    """What the SENSE phase produced: a normalized view of the world."""

    iteration: int
    inputs: Dict[str, Any]
    signals: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """What the REASON phase produced: an intended action + rationale."""

    action: str
    arguments: Dict[str, Any]
    rationale: str
    confidence: float
    idempotency_key: str


@dataclass(frozen=True)
class ActionResult:
    """What the ACT phase produced: the outcome of executing a Decision."""

    action: str
    ok: bool
    output: Dict[str, Any]
    error: Optional[str] = None


@dataclass(frozen=True)
class Reflection:
    """What the REFLECT phase produced: progress assessment + next setpoint."""

    converged: bool
    score: float
    feedback: str
    next_hint: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopState:
    """Mutable carrier threaded through the whole loop.

    `score` is the control-theory 'process variable' we drive toward `setpoint`.
    `history` keeps a compact, serializable record for replay and audit.
    """

    goal: str
    setpoint: float
    score: float = 0.0
    iteration: int = 0
    memory: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def error_signal(self) -> float:
        """Distance from the target. Loops minimize this."""
        return max(0.0, self.setpoint - self.score)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "setpoint": self.setpoint,
            "score": self.score,
            "iteration": self.iteration,
            "memory": dict(self.memory),
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)


def serialize(obj: Any) -> Dict[str, Any]:
    """Best-effort serialization for dataclasses used in tracing."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return {"value": obj}
