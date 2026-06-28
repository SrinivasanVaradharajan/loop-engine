"""Deterministic tracing + replay.

Determinism principle: if you can't replay it, you can't trust it. Every phase
emits a structured event. The ordered event log IS the proof of what the loop
did, and `replay_trace` lets you reconstruct the score trajectory offline —
in CI, in a postmortem, or in an audit — without re-running side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json
import hashlib


@dataclass
class TraceEvent:
    seq: int
    phase: str
    payload: Dict[str, Any]


@dataclass
class TraceRecorder:
    events: List[TraceEvent] = field(default_factory=list)

    def record(self, phase: str, payload: Dict[str, Any]) -> None:
        self.events.append(TraceEvent(seq=len(self.events), phase=phase, payload=_safe(payload)))

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps({"seq": e.seq, "phase": e.phase, "payload": e.payload}, sort_keys=True)
            for e in self.events
        )

    def digest(self) -> str:
        """Stable hash of the whole trace. Identical inputs => identical digest.

        This is how you assert determinism in tests: run twice, compare digests.
        """
        return hashlib.sha256(self.to_jsonl().encode("utf-8")).hexdigest()


def replay_trace(jsonl: str) -> Dict[str, Any]:
    """Reconstruct the score trajectory + stop reason from a recorded trace."""
    scores: List[float] = []
    stop_reason = "unknown"
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        evt = json.loads(line)
        if evt["phase"] == "reflect":
            scores.append(evt["payload"]["reflection"]["score"])
        elif evt["phase"] in {"budget", "oscillation", "guardrail"}:
            stop_reason = evt["phase"]
    return {
        "iterations": len(scores),
        "scores": scores,
        "converged": bool(scores) and scores[-1] >= scores[0],
        "stop_reason": stop_reason,
    }


def _safe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Round-trip through JSON so traces are guaranteed serializable."""
    return json.loads(json.dumps(payload, default=lambda o: getattr(o, "__dict__", str(o))))
