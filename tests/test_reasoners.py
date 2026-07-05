"""Tests proving the seam holds across Rule, LLM, and Crew reasoners.

The controller (budgets, guardrails, idempotency, replay) is unchanged; only the
REASON phase varies. We assert that every reasoner:
  * converges to the SAME validated record in the SAME iteration count, and
  * is itself deterministic (run twice -> byte-identical trace digest).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loopengine import (
    LoopState, LoopController, LoopConfig, StopReason,
    DeterministicLLMClient, LLMReasoner, CrewReasoner, crewai_available,
    stable_key,
)
from loopengine.guardrails import Budget, default_guardrails
from loopengine.phases import Agent
from examples.self_correcting_extractor import (
    RAW, REQUIRED, Sensor, RuleReasoner, Actuator, Reflector,
)

EXPECTED_RECORD = {
    "customer_id": "88421",
    "amount": 1250.0,
    "currency": "USD",
    "intent": "refund",
}


def _extract(field):
    return RuleReasoner()._extract(field)


def _state():
    return LoopState(goal="extract a valid service record", setpoint=1.0)


def _build(reasoner):
    agent = Agent(Sensor(), reasoner, Actuator(), Reflector())
    config = LoopConfig(
        budget=Budget(max_iterations=8, max_seconds=10, max_cost_units=50),
        guardrails=default_guardrails(agent.actions),
        patience=3,
    )
    return LoopController(agent, config)


def _run(reasoner):
    state = _state()
    controller = _build(reasoner)
    outcome = controller.run(state)
    return state, outcome, controller


# --------------------------------------------------------------------------- #
# LLM reasoner
# --------------------------------------------------------------------------- #
def test_llm_reasoner_converges_like_rules():
    state, outcome, _ = _run(LLMReasoner(DeterministicLLMClient(_extract)))
    assert outcome.converged is True
    assert outcome.stop_reason == StopReason.CONVERGED
    assert outcome.iterations == 4
    assert state.memory.get("record") == EXPECTED_RECORD


def test_llm_reasoner_is_deterministic():
    _, _, c1 = _run(LLMReasoner(DeterministicLLMClient(_extract)))
    _, _, c2 = _run(LLMReasoner(DeterministicLLMClient(_extract)))
    assert c1.recorder.digest() == c2.recorder.digest()


# --------------------------------------------------------------------------- #
# Crew reasoner
# --------------------------------------------------------------------------- #
def test_crew_reasoner_converges_like_rules():
    state, outcome, _ = _run(CrewReasoner(_extract, use_crewai=False))
    assert outcome.converged is True
    assert outcome.stop_reason == StopReason.CONVERGED
    assert outcome.iterations == 4
    assert state.memory.get("record") == EXPECTED_RECORD


def test_crew_reasoner_is_deterministic():
    _, _, c1 = _run(CrewReasoner(_extract, use_crewai=False))
    _, _, c2 = _run(CrewReasoner(_extract, use_crewai=False))
    assert c1.recorder.digest() == c2.recorder.digest()


def test_crew_falls_back_to_local_when_crewai_absent():
    # With crewai not installed, use_crewai=True must degrade gracefully.
    reasoner = CrewReasoner(_extract, use_crewai=True)
    assert reasoner.use_crewai == crewai_available()
    state, outcome, _ = _run(reasoner)
    assert outcome.converged is True
    assert state.memory.get("record") == EXPECTED_RECORD


# --------------------------------------------------------------------------- #
# The seam: all three reasoners agree on the final record
# --------------------------------------------------------------------------- #
def test_all_reasoners_agree_on_record():
    records = []
    for reasoner in (
        RuleReasoner(),
        LLMReasoner(DeterministicLLMClient(_extract)),
        CrewReasoner(_extract, use_crewai=False),
    ):
        state, _, _ = _run(reasoner)
        records.append(state.memory.get("record"))
    assert all(r == EXPECTED_RECORD for r in records)


def test_stable_key_is_content_addressed():
    k1 = stable_key("set_field", {"field": "currency", "value": "USD"})
    k2 = stable_key("set_field", {"value": "USD", "field": "currency"})  # order-independent
    assert k1 == k2
    assert k1.startswith("set_field:")


if __name__ == "__main__":
    test_llm_reasoner_converges_like_rules()
    test_llm_reasoner_is_deterministic()
    test_crew_reasoner_converges_like_rules()
    test_crew_reasoner_is_deterministic()
    test_crew_falls_back_to_local_when_crewai_absent()
    test_all_reasoners_agree_on_record()
    test_stable_key_is_content_addressed()
    print("all reasoner tests passed")
