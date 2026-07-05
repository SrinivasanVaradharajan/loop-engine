"""Reusable REASON-phase implementations behind one stable seam.

The `Reasoner` protocol (see ``phases.py``) is the only seam you change to move
from hand-written rules to a single LLM call to a whole crew of collaborating
agents. Every reasoner here honours the identical contract::

    reason(state: LoopState, obs: Observation) -> Decision

so the ``LoopController`` — budgets, guardrails, idempotency, replay — never
learns which brain is behind the seam. That separation is the whole point:
*let the model (or the crew) be creative; keep the loop deterministic.*

This module is dependency-free. The LLM and Crew reasoners run **offline and
reproducibly** by default (via :class:`DeterministicLLMClient` and a local
crew), and expose the exact shape you would wire a real model / CrewAI crew
into for production.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, Callable, Dict, List, Optional, Protocol

from .state import LoopState, Observation, Decision


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def stable_key(action: str, arguments: Dict[str, Any]) -> str:
    """A content-addressed idempotency key.

    Identical (action, arguments) always yield the same key, so the controller
    can safely skip duplicate side effects on retries — regardless of which
    reasoner produced the decision.
    """
    blob = json.dumps(arguments, sort_keys=True, default=str)
    return f"{action}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]}"


# Type alias: given a field name, return its extracted value (or None).
ExtractFn = Callable[[str], Any]


# --------------------------------------------------------------------------- #
# LLM reasoner
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    """The one method a language-model backend must provide.

    Swap in OpenAI, Bedrock, Vertex, Azure, a local vLLM server, etc. — anything
    that turns a prompt string into a completion string honours this contract.
    """

    def complete(self, prompt: str) -> str: ...


class DeterministicLLMClient:
    """A stand-in 'model' that returns a JSON action deterministically.

    It lets you exercise :class:`LLMReasoner` with **no API key, no network, and
    a reproducible trace** — ideal for tests, CI, demos, and this article's
    worked example. It mimics a well-behaved model: read the target field off
    the prompt, "reason", and emit a strict-JSON action.

    Replace it with a real :class:`LLMClient` and the reasoner code is unchanged.
    """

    def __init__(self, extract: ExtractFn):
        self._extract = extract

    def complete(self, prompt: str) -> str:
        # The default prompt encodes the target field on its final line:
        #   "FIELD: <name>"
        field = prompt.strip().splitlines()[-1].split("FIELD:")[-1].strip()
        value = self._extract(field)
        return json.dumps(
            {
                "action": "set_field",
                "arguments": {"field": field, "value": value},
                "confidence": 0.9 if value is not None else 0.2,
                "rationale": f"model extracted {value!r} for field '{field}'",
            },
            sort_keys=True,
        )


class LLMReasoner:
    """REASON via a single model call, parsed into a typed :class:`Decision`.

    The controller sees exactly what it saw with the rule-based reasoner: an
    ``Observation`` in, a ``Decision`` out. Only the *inside* of REASON changed.
    """

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: Optional[Callable[[LoopState, Observation], str]] = None,
    ):
        self.client = client
        self.prompt_builder = prompt_builder or self._default_prompt

    @staticmethod
    def _default_prompt(state: LoopState, obs: Observation) -> str:
        field = obs.signals["missing"][0]
        raw = obs.inputs.get("raw", "")
        return (
            "You are a precise extraction agent. Read RAW and return a single "
            "JSON action of the form "
            '{"action":"set_field","arguments":{"field":..,"value":..},'
            '"confidence":..,"rationale":..}.\n'
            f"RAW: {raw}\n"
            f"FIELD: {field}"
        )

    def reason(self, state: LoopState, obs: Observation) -> Decision:
        raw = self.client.complete(self.prompt_builder(state, obs))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # defensive: a real model can misfire
            raise ValueError(f"LLM returned non-JSON output: {raw!r}") from exc

        action = data["action"]
        args = data["arguments"]
        return Decision(
            action=action,
            arguments=args,
            rationale=data.get("rationale", ""),
            confidence=float(data.get("confidence", 0.5)),
            idempotency_key=stable_key(action, args),
        )


# --------------------------------------------------------------------------- #
# Crew reasoner
# --------------------------------------------------------------------------- #
def crewai_available() -> bool:
    """True when the optional `crewai` package is importable."""
    try:
        import crewai  # noqa: F401
        return True
    except Exception:
        return False


class CrewReasoner:
    """REASON by running a small crew of role-specialized agents in sequence.

    A crew does **not** replace the loop — it lives *inside* the REASON phase.
    Here two roles collaborate: an **Extractor** proposes a value and an
    **Auditor** validates and scores it. The crew still returns exactly one
    :class:`Decision`, so the controller never knows a team sits behind the seam.

    Two execution paths, one contract:

    * **Local crew (default)** — a deterministic, dependency-free pipeline with
      the same extractor→auditor shape. Runs anywhere, reproducibly.
    * **Real CrewAI** — set ``use_crewai=True`` (requires ``crewai`` installed
      and a configured LLM). :meth:`_build_crewai_crew` shows the real API you
      would use in production.

    The five controls scale with the crew:
      * budgets  -> ``max_rpm`` / delegation depth (crew-level ceilings)
      * stop     -> task-completion criteria per agent
      * guardrails -> validate every inter-agent handoff, not just the final answer
      * idempotency -> content-addressed keys keep retried tool calls safe
      * replay   -> log each agent's input/output so any run can be reconstructed
    """

    def __init__(
        self,
        extract: ExtractFn,
        use_crewai: bool = False,
        max_rpm: int = 10,
    ):
        self.extract = extract
        self.max_rpm = max_rpm
        self.use_crewai = bool(use_crewai and crewai_available())

    # -- public contract ---------------------------------------------------- #
    def reason(self, state: LoopState, obs: Observation) -> Decision:
        field: str = obs.signals["missing"][0]
        raw: str = obs.inputs.get("raw", "")

        if self.use_crewai:
            value, score = self._run_crewai_crew(field, raw)
        else:
            value, score = self._run_local_crew(field, raw)

        args = {"field": field, "value": value}
        return Decision(
            action="set_field",
            arguments=args,
            rationale=(
                f"crew(extractor->auditor) resolved '{field}'={value!r} "
                f"with audit_score={score}"
            ),
            confidence=score,
            idempotency_key=stable_key("set_field", args),
        )

    # -- deterministic local crew (default) --------------------------------- #
    def _run_local_crew(self, field: str, raw: str) -> tuple[Any, float]:
        """Extractor agent proposes; Auditor agent validates + scores."""
        value = self._extractor_agent(field, raw)
        score = self._auditor_agent(field, value)
        return value, score

    def _extractor_agent(self, field: str, raw: str) -> Any:
        return self.extract(field)

    def _auditor_agent(self, field: str, value: Any) -> float:
        # A minimal, deterministic "review": present + non-empty scores high.
        if value is None or value == "":
            return 0.2
        return 0.9

    # -- real CrewAI path (opt-in) ------------------------------------------ #
    def _build_crewai_crew(self, field: str, raw: str):
        """Construct a real CrewAI crew (extractor + auditor).

        Illustrates the production API. `kickoff()` needs a configured LLM, so
        this path is opt-in; the default local crew keeps the repo runnable and
        deterministic everywhere.
        """
        from crewai import Agent, Task, Crew, Process  # type: ignore

        extractor = Agent(
            role="Extractor",
            goal="Pull the requested structured field from raw text",
            backstory="A meticulous parser that returns only the value asked for.",
            allow_delegation=False,
        )
        auditor = Agent(
            role="Auditor",
            goal="Validate the extracted value and return a confidence score 0-1",
            backstory="A strict reviewer that rejects malformed or empty values.",
            allow_delegation=False,
        )
        extract_task = Task(
            description=f"Extract the '{field}' field from: {raw}",
            expected_output="The raw value of the field, or the string 'None'.",
            agent=extractor,
        )
        audit_task = Task(
            description="Validate the extracted value; respond with a score in [0,1].",
            expected_output="A single float between 0 and 1.",
            agent=auditor,
        )
        return Crew(
            agents=[extractor, auditor],
            tasks=[extract_task, audit_task],
            process=Process.sequential,  # deterministic handoff order
            max_rpm=self.max_rpm,        # crew-level budget
        )

    def _run_crewai_crew(self, field: str, raw: str) -> tuple[Any, float]:
        crew = self._build_crewai_crew(field, raw)
        result = crew.kickoff(inputs={"field": field, "raw": raw})
        # A real crew returns free-form/structured output; we defensively parse.
        payload = getattr(result, "json_dict", None) or {}
        value = payload.get("value", self.extract(field))
        score = float(payload.get("score", 0.9 if value is not None else 0.2))
        return value, score
