"""loopengine — a deterministic agentic-loop reference implementation.

Anatomy of an agentic loop: Sense -> Reason -> Act -> Reflect, wrapped in
enterprise determinism controls (budgets, stop conditions, guardrails,
idempotency, and deterministic replay).
"""

from .state import LoopState, Observation, Decision, ActionResult, Reflection
from .controller import LoopController, LoopConfig, StopReason
from .guardrails import Guardrail, GuardrailError, BudgetExceeded
from .replay import TraceRecorder, replay_trace
from .reasoners import (
    LLMClient,
    DeterministicLLMClient,
    LLMReasoner,
    CrewReasoner,
    crewai_available,
    stable_key,
)

__all__ = [
    "LoopState",
    "Observation",
    "Decision",
    "ActionResult",
    "Reflection",
    "LoopController",
    "LoopConfig",
    "StopReason",
    "Guardrail",
    "GuardrailError",
    "BudgetExceeded",
    "TraceRecorder",
    "replay_trace",
    "LLMClient",
    "DeterministicLLMClient",
    "LLMReasoner",
    "CrewReasoner",
    "crewai_available",
    "stable_key",
]

__version__ = "0.1.0"
