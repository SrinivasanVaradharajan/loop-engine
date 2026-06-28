"""Worked example: a self-correcting structured-extraction loop.

Goal: turn a messy free-text service request into a validated structured record
{customer_id, amount, currency, intent}. No external LLM is used so the example
is fully deterministic and runs anywhere — but the SHAPE is exactly what you'd
wire a real model into (swap `RuleReasoner` for an LLM-backed reasoner).

The loop fixes ONE field per iteration (Reason), applies it (Act), and scores
completeness/validity (Reflect). It converges when the record is fully valid,
or stops on budget — demonstrating determinism around autonomous behavior.

Run:  python -m examples.self_correcting_extractor
"""

from __future__ import annotations

import re
from typing import Dict, List

from loopengine import (
    LoopState, Observation, Decision, ActionResult, Reflection,
    LoopController, LoopConfig,
)
from loopengine.guardrails import Budget, default_guardrails
from loopengine.phases import Agent

RAW = "Req#  cust=88421 wants a REFUND of  1,250.00 usd  pls"

REQUIRED = ["customer_id", "amount", "currency", "intent"]
VALIDATORS = {
    "customer_id": lambda v: bool(re.fullmatch(r"\d{5}", str(v))),
    "amount": lambda v: isinstance(v, (int, float)) and v > 0,
    "currency": lambda v: v in {"USD", "EUR", "GBP"},
    "intent": lambda v: v in {"refund", "dispute", "inquiry"},
}


class Sensor:
    """SENSE: expose the current record + which required fields are missing."""

    def sense(self, state: LoopState) -> Observation:
        record = state.memory.get("record", {})
        missing = [f for f in REQUIRED if f not in record or not VALIDATORS[f](record.get(f))]
        return Observation(iteration=state.iteration, inputs={"raw": RAW},
                           signals={"missing": missing, "record": dict(record)})


class RuleReasoner:
    """REASON: pick the next missing field and propose an extraction.

    A real system would call an LLM here. The contract is identical: input is an
    Observation, output is a typed Decision with an idempotency key.
    """

    def _extract(self, field: str) -> object:
        if field == "customer_id":
            m = re.search(r"cust=(\d+)", RAW); return m.group(1) if m else None
        if field == "amount":
            m = re.search(r"([\d,]+\.\d{2})", RAW); return float(m.group(1).replace(",", "")) if m else None
        if field == "currency":
            m = re.search(r"\b(usd|eur|gbp)\b", RAW, re.I); return m.group(1).upper() if m else None
        if field == "intent":
            m = re.search(r"\b(refund|dispute|inquiry)\b", RAW, re.I); return m.group(1).lower() if m else None
        return None

    def reason(self, state: LoopState, obs: Observation) -> Decision:
        missing: List[str] = obs.signals["missing"]
        field = missing[0]
        value = self._extract(field)
        return Decision(
            action="set_field",
            arguments={"field": field, "value": value},
            rationale=f"field '{field}' missing/invalid; extracted {value!r} from raw text",
            confidence=0.9 if value is not None else 0.2,
            idempotency_key=f"set:{field}:{value}",
        )


class Actuator:
    """ACT: apply the decision to the working record (the side effect)."""

    actions = ["set_field"]

    def act(self, state: LoopState, decision: Decision) -> ActionResult:
        record: Dict = state.memory.setdefault("record", {})
        field = decision.arguments["field"]
        value = decision.arguments["value"]
        if value is None:
            return ActionResult(action=decision.action, ok=False, output={}, error=f"no value for {field}")
        record[field] = value
        return ActionResult(action=decision.action, ok=True, output={"record": dict(record)})


class Reflector:
    """REFLECT: score completeness/validity and decide convergence."""

    def reflect(self, state: LoopState, result: ActionResult) -> Reflection:
        record = state.memory.get("record", {})
        valid = sum(1 for f in REQUIRED if f in record and VALIDATORS[f](record[f]))
        score = valid / len(REQUIRED)
        converged = score >= 1.0
        return Reflection(
            converged=converged,
            score=score,
            feedback=f"{valid}/{len(REQUIRED)} fields valid",
            next_hint={"valid_fields": valid},
        )


def build_controller() -> LoopController:
    agent = Agent(Sensor(), RuleReasoner(), Actuator(), Reflector())
    config = LoopConfig(
        budget=Budget(max_iterations=8, max_seconds=10, max_cost_units=50),
        guardrails=default_guardrails(agent.actions),
        patience=3,
    )
    return LoopController(agent, config)


def main() -> None:
    state = LoopState(goal="extract a valid service record", setpoint=1.0)
    controller = build_controller()
    outcome = controller.run(state)

    print("RAW INPUT :", RAW)
    print("RECORD    :", state.memory.get("record"))
    print("ITERATIONS:", outcome.iterations)
    print("STOP      :", outcome.stop_reason.value)
    print("CONVERGED :", outcome.converged)
    print("TRACE HASH:", controller.recorder.digest()[:16], "(stable across runs)")


if __name__ == "__main__":
    main()
