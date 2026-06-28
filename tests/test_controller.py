"""Tests proving the loop is deterministic, bounded, and convergent."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loopengine import LoopState, LoopController, LoopConfig, StopReason
from loopengine.guardrails import Budget, Guardrail
from loopengine.phases import Agent
from examples.self_correcting_extractor import (
    Sensor, RuleReasoner, Actuator, Reflector, build_controller,
)


def _state():
    return LoopState(goal="extract a valid service record", setpoint=1.0)


def test_loop_converges():
    outcome = build_controller().run(_state())
    assert outcome.converged is True
    assert outcome.stop_reason == StopReason.CONVERGED
    assert outcome.final_score == 1.0
    assert outcome.iterations <= 8


def test_determinism_same_trace_hash():
    """Same inputs must produce a byte-identical trace — the determinism proof."""
    c1 = build_controller(); c1.run(_state())
    c2 = build_controller(); c2.run(_state())
    assert c1.recorder.digest() == c2.recorder.digest()


def test_budget_is_a_hard_stop():
    agent = Agent(Sensor(), RuleReasoner(), Actuator(), Reflector())
    # One iteration can't fill four fields, so the budget must stop it.
    config = LoopConfig(budget=Budget(max_iterations=1, max_seconds=5, max_cost_units=10))
    outcome = LoopController(agent, config).run(_state())
    assert outcome.iterations <= 1
    assert outcome.converged is False
    assert outcome.stop_reason == StopReason.BUDGET


def test_guardrail_blocks_unapproved_action():
    agent = Agent(Sensor(), RuleReasoner(), Actuator(), Reflector())
    deny_all = Guardrail(name="deny", rule=lambda a, args: False, message="blocked")
    config = LoopConfig(budget=Budget(max_iterations=8), guardrails=[deny_all])
    outcome = LoopController(agent, config).run(_state())
    assert outcome.stop_reason == StopReason.GUARDRAIL
    assert outcome.converged is False


if __name__ == "__main__":
    test_loop_converges()
    test_determinism_same_trace_hash()
    test_budget_is_a_hard_stop()
    test_guardrail_blocks_unapproved_action()
    print("all tests passed")
