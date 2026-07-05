"""Worked example: one seam, three reasoners (Rule -> LLM -> Crew).

This is the executable version of the article's "One Seam to Swap" idea. We run
the *same* self-correcting extraction loop three times, changing only the REASON
phase:

    1. RuleReasoner  — hand-written regex (deterministic demo)
    2. LLMReasoner   — a single model call (offline, deterministic stub client)
    3. CrewReasoner  — a crew of agents (Extractor -> Auditor) inside REASON

The controller, budgets, guardrails, idempotency, and replay never change. All
three converge to the *same* validated record in the same number of iterations —
proving the seam holds. Each reasoner is also internally deterministic: run it
twice and the trace digest is byte-identical.

Run:  python -m examples.reasoner_swap
"""

from __future__ import annotations

from typing import Any, Tuple

from loopengine import (
    LoopState, LoopController, LoopConfig,
    DeterministicLLMClient, LLMReasoner, CrewReasoner, crewai_available,
)
from loopengine.guardrails import Budget, default_guardrails
from loopengine.phases import Agent

# Reuse the domain (Sensor/Actuator/Reflector + the extraction) from the
# original worked example so the ONLY thing that varies is the Reasoner.
from examples.self_correcting_extractor import (
    RAW, REQUIRED, Sensor, RuleReasoner, Actuator, Reflector,
)


def _extract(field: str) -> Any:
    """Deterministic field extraction shared by the LLM stub + the crew."""
    return RuleReasoner()._extract(field)


def _build(reasoner) -> LoopController:
    agent = Agent(Sensor(), reasoner, Actuator(), Reflector())
    config = LoopConfig(
        budget=Budget(max_iterations=8, max_seconds=10, max_cost_units=50),
        guardrails=default_guardrails(agent.actions),
        patience=3,
    )
    return LoopController(agent, config)


def _run(reasoner) -> Tuple[dict, int, str, bool, str]:
    state = LoopState(goal="extract a valid service record", setpoint=1.0)
    controller = _build(reasoner)
    outcome = controller.run(state)
    return (
        state.memory.get("record", {}),
        outcome.iterations,
        outcome.stop_reason.value,
        outcome.converged,
        controller.recorder.digest()[:16],
    )


def main() -> None:
    reasoners = {
        "RuleReasoner": RuleReasoner(),
        "LLMReasoner": LLMReasoner(DeterministicLLMClient(_extract)),
        "CrewReasoner": CrewReasoner(_extract, use_crewai=False),
    }

    print("RAW INPUT :", RAW)
    print(f"CrewAI installed: {crewai_available()} "
          "(CrewReasoner uses its deterministic local crew when False)\n")

    print(f"{'REASONER':<14} {'ITERS':<6} {'STOP':<11} {'CONVERGED':<10} TRACE(16)")
    print("-" * 68)
    records = {}
    for name, reasoner in reasoners.items():
        record, iters, stop, converged, digest = _run(reasoner)
        records[name] = record
        print(f"{name:<14} {iters:<6} {stop:<11} {str(converged):<10} {digest}")

    # The seam holds: every reasoner produces the SAME validated record.
    baseline = records["RuleReasoner"]
    same = all(records[n] == baseline for n in records)
    print("\nFINAL RECORD :", baseline)
    print("ALL REASONERS AGREE:", same, f"({'/'.join(REQUIRED)})")


if __name__ == "__main__":
    main()
