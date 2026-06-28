"""The four phases as explicit, swappable protocols.

Anatomy:
  SENSE   -> gather + normalize inputs into an Observation
  REASON  -> turn an Observation into a Decision (intended action)
  ACT     -> execute the Decision, returning an ActionResult
  REFLECT -> score progress, decide convergence, emit a next-iteration hint

Determinism principle: each phase is a pure-ish function of explicit state.
Swap a phase implementation (e.g. a real LLM reasoner) without touching the
controller, budgets, guardrails, or replay machinery.
"""

from __future__ import annotations

from typing import Protocol, List
from .state import LoopState, Observation, Decision, ActionResult, Reflection


class Sensor(Protocol):
    def sense(self, state: LoopState) -> Observation: ...


class Reasoner(Protocol):
    def reason(self, state: LoopState, obs: Observation) -> Decision: ...


class Actuator(Protocol):
    def act(self, state: LoopState, decision: Decision) -> ActionResult: ...


class Reflector(Protocol):
    def reflect(self, state: LoopState, result: ActionResult) -> Reflection: ...


class Agent:
    """Bundle of the four phase implementations the controller will drive."""

    def __init__(self, sensor: Sensor, reasoner: Reasoner, actuator: Actuator, reflector: Reflector):
        self.sensor = sensor
        self.reasoner = reasoner
        self.actuator = actuator
        self.reflector = reflector

    @property
    def actions(self) -> List[str]:
        return getattr(self.actuator, "actions", [])
