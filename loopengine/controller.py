"""The LoopController — the deterministic engine that drives the four phases.

This is where 'Determinism meets Autonomy': the agent's phases may be creative,
but the *control* around them is strict and reproducible:

  * fixed phase order, one decision per iteration
  * budgets enforced before every iteration (the primary stop condition)
  * guardrails enforced before every action
  * idempotency keys prevent duplicate side effects on retried actions
  * convergence + oscillation detection decide when to stop *early*
  * every step is recorded for exact replay / audit
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .state import LoopState
from .phases import Agent
from .guardrails import Guardrail, Budget, GuardrailError, BudgetExceeded, default_guardrails
from .replay import TraceRecorder


class StopReason(str, Enum):
    CONVERGED = "converged"
    BUDGET = "budget_exhausted"
    OSCILLATION = "oscillation_detected"
    GUARDRAIL = "guardrail_blocked"
    ERROR = "unrecoverable_error"


@dataclass
class LoopConfig:
    budget: Budget
    guardrails: Optional[List[Guardrail]] = None
    cost_per_iteration: float = 1.0
    # Stop early if the error signal fails to improve for N iterations.
    patience: int = 2
    min_improvement: float = 1e-6


@dataclass
class LoopOutcome:
    stop_reason: StopReason
    iterations: int
    final_score: float
    converged: bool
    state: LoopState


class LoopController:
    """Drives an Agent to convergence under strict, reproducible controls."""

    def __init__(self, agent: Agent, config: LoopConfig, recorder: Optional[TraceRecorder] = None):
        self.agent = agent
        self.config = config
        self.recorder = recorder or TraceRecorder()
        self.guardrails = config.guardrails or default_guardrails(agent.actions)
        self._seen_keys: set[str] = set()

    def run(self, state: LoopState) -> LoopOutcome:
        cfg = self.config
        best_error = float("inf")
        stale = 0
        stop = StopReason.BUDGET

        while True:
            # --- Stop condition #1: budget (checked BEFORE any work) ---
            try:
                cfg.budget.check_iteration(state.iteration)
            except BudgetExceeded as exc:
                self.recorder.record("budget", {"detail": str(exc)})
                stop = StopReason.BUDGET
                break

            state.iteration += 1

            # --- SENSE ---
            obs = self.agent.sensor.sense(state)
            self.recorder.record("sense", {"observation": obs.__dict__})

            # --- REASON ---
            decision = self.agent.reasoner.reason(state, obs)
            self.recorder.record("reason", {"decision": decision.__dict__})

            # --- Guardrails (before side effects) ---
            try:
                for g in self.guardrails:
                    g.assert_allowed(decision.action, decision.arguments)
            except GuardrailError as exc:
                self.recorder.record("guardrail", {"detail": str(exc)})
                stop = StopReason.GUARDRAIL
                break

            # --- Idempotency: skip duplicate side effects on identical retries ---
            if decision.idempotency_key in self._seen_keys:
                self.recorder.record("idempotent_skip", {"key": decision.idempotency_key})
            else:
                self._seen_keys.add(decision.idempotency_key)
                cfg.budget.charge(cfg.cost_per_iteration)

                # --- ACT ---
                result = self.agent.actuator.act(state, decision)
                self.recorder.record("act", {"result": result.__dict__})

                if not result.ok:
                    # Recoverable errors feed back into REFLECT; here we let the
                    # reflector decide whether to continue or give up.
                    self.recorder.record("act_error", {"error": result.error})

                # --- REFLECT ---
                reflection = self.agent.reflector.reflect(state, result)
                self.recorder.record("reflect", {"reflection": reflection.__dict__})

                state.score = reflection.score
                state.memory.update(reflection.next_hint)
                state.history.append(state.snapshot())

                # --- Stop condition #2: convergence ---
                if reflection.converged or state.score >= state.setpoint:
                    stop = StopReason.CONVERGED
                    break

                # --- Stop condition #3: oscillation / no-progress (patience) ---
                err = state.error_signal()
                if best_error - err > cfg.min_improvement:
                    best_error = err
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        self.recorder.record("oscillation", {"stale_for": stale})
                        stop = StopReason.OSCILLATION
                        break

        return LoopOutcome(
            stop_reason=stop,
            iterations=state.iteration,
            final_score=state.score,
            converged=(stop == StopReason.CONVERGED),
            state=state,
        )
