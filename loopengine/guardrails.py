"""Guardrails and budgets — the controls that make autonomy predictable.

Determinism principle: autonomy without bounds is non-determinism in disguise.
Budgets cap *how long* a loop may run; guardrails cap *what* it may do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List
import time


class GuardrailError(Exception):
    """Raised when a proposed action violates a policy guardrail."""


class BudgetExceeded(Exception):
    """Raised when the loop exhausts an iteration / time / cost budget."""


@dataclass
class Budget:
    """Hard stop conditions. The single most important determinism control."""

    max_iterations: int = 8
    max_seconds: float = 30.0
    max_cost_units: float = 100.0

    _spent_cost: float = field(default=0.0, init=False)
    _start: float = field(default_factory=time.monotonic, init=False)

    def charge(self, cost_units: float) -> None:
        self._spent_cost += cost_units
        if self._spent_cost > self.max_cost_units:
            raise BudgetExceeded(
                f"cost budget exhausted: {self._spent_cost:.1f} > {self.max_cost_units:.1f}"
            )

    def check_iteration(self, iteration: int) -> None:
        if iteration >= self.max_iterations:
            raise BudgetExceeded(f"iteration budget exhausted: {iteration} >= {self.max_iterations}")
        if (time.monotonic() - self._start) > self.max_seconds:
            raise BudgetExceeded("wall-clock budget exhausted")


@dataclass
class Guardrail:
    """A named policy predicate evaluated against a proposed action.

    Each rule returns True when the action is ALLOWED. A failing rule blocks
    the action deterministically — same input, same verdict, every time.
    """

    name: str
    rule: Callable[[str, Dict], bool]
    message: str = "action blocked by guardrail"

    def assert_allowed(self, action: str, arguments: Dict) -> None:
        if not self.rule(action, arguments):
            raise GuardrailError(f"[{self.name}] {self.message}")


def default_guardrails(allowed_actions: List[str]) -> List[Guardrail]:
    """A sensible starting policy set for enterprise loops."""
    allow = set(allowed_actions)
    return [
        Guardrail(
            name="allowlist",
            rule=lambda action, args: action in allow,
            message="action is not on the approved allowlist",
        ),
        Guardrail(
            name="no-empty-args",
            rule=lambda action, args: isinstance(args, dict),
            message="action arguments must be a structured object",
        ),
    ]
