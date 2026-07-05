# loop-engine

> **Determinism Meets Autonomy** — a tiny, dependency-free reference implementation of an
> agentic loop: **Sense → Reason → Act → Reflect**, wrapped in the controls that make
> autonomous behavior *predictable* enough for production.

This is the companion code for the article
**"Determinism Meets Autonomy: The Anatomy of an Agentic Loop (Sense → Reason → Act → Reflect)."**

Most "agents" are a single model call. Real autonomy is a **loop**: the agent observes,
decides, acts, and grades its own progress — repeating until it converges or runs out of
budget. The hard part isn't the creativity inside the loop; it's the **engineering around
it** that keeps the loop bounded, auditable, and reproducible.

---

## The anatomy

```
            ┌─────────────────────────────────────────────┐
            │                 LoopController              │
            │   (budgets · guardrails · stop conditions)  │
            │                                             │
   state →  │   SENSE → REASON → ACT → REFLECT            │ → outcome
            │     ▲                        │              │
            │     └──────── next hint ◀────┘              │
            └─────────────────────────────────────────────┘
                         every step recorded → replay
```

| Phase | Question it answers | In this repo |
|-------|--------------------|--------------|
| **Sense** | "What is true right now?" | `Sensor.sense()` → `Observation` |
| **Reason** | "What should I do next?" | `RuleReasoner` / `LLMReasoner` / `CrewReasoner` → `Decision` |
| **Act** | "Do it (with guardrails)." | `Actuator.act()` → `ActionResult` |
| **Reflect** | "Did it help? Am I done?" | `Reflector.reflect()` → `Reflection` |

## The determinism controls

Autonomy without bounds is just non-determinism. `loop-engine` ships the five controls every
production loop needs:

1. **Budgets** — `max_iterations`, `max_seconds`, `max_cost_units` (the primary stop condition).
2. **Stop conditions** — convergence, oscillation/no-progress (`patience`), guardrail block.
3. **Guardrails** — policy predicates evaluated *before* any side effect.
4. **Idempotency** — identical decisions never fire the same side effect twice.
5. **Deterministic replay** — every phase is recorded; a stable SHA-256 `digest()` proves
   identical inputs produce identical runs.

---

## Quickstart

```bash
git clone https://github.com/SrinivasanVaradharajan/loop-engine.git
cd loop-engine
python -m examples.self_correcting_extractor
```

Expected output:

```
RAW INPUT : Req#  cust=88421 wants a REFUND of  1,250.00 usd  pls
RECORD    : {'customer_id': '88421', 'amount': 1250.0, 'currency': 'USD', 'intent': 'refund'}
ITERATIONS: 4
STOP      : converged
CONVERGED : True
TRACE HASH: <stable 16-char hash> (stable across runs)
```

## Run the tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite proves the properties that matter:

- `test_loop_converges` — the loop reaches a fully valid record.
- `test_determinism_same_trace_hash` — two runs produce a **byte-identical** trace.
- `test_budget_is_a_hard_stop` — autonomy is bounded by budget.
- `test_guardrail_blocks_unapproved_action` — policy halts the loop safely.
- `test_reasoners.py` — Rule, LLM, and Crew reasoners all converge to the **same
  record** in the same iterations, each internally deterministic (the seam holds).

---

## Three reasoners, one seam

The `Reasoner` protocol is the **only** seam you change to evolve the brain of the
loop. `loop-engine` ships three implementations of it — all honouring the identical
contract `reason(state, obs) -> Decision`, so the controller never knows which one
is behind the seam:

| Reasoner | Inside REASON | When to reach for it |
|----------|---------------|----------------------|
| `RuleReasoner` | hand-written regex | deterministic baseline / cheapest path |
| `LLMReasoner` | one model call → parsed JSON | open-ended reasoning, still one shot |
| `CrewReasoner` | a crew of agents (Extractor → Auditor) | work that needs specialists collaborating |

Run all three on the same loop and watch the seam hold — identical record, identical
iteration count:

```bash
python -m examples.reasoner_swap
```

```
REASONER       ITERS  STOP        CONVERGED  TRACE(16)
--------------------------------------------------------------------
RuleReasoner   4      converged   True       59300832edab42a4
LLMReasoner    4      converged   True       53194adddb7c3233
CrewReasoner   4      converged   True       c8aacb0db41f14c2

FINAL RECORD : {'customer_id': '88421', 'amount': 1250.0, 'currency': 'USD', 'intent': 'refund'}
ALL REASONERS AGREE: True (customer_id/amount/currency/intent)
```

Each reasoner is internally deterministic (same reasoner, two runs → same digest);
different reasoners agree on the *outcome* while narrating it differently.

### 1 · LLM reasoner

`LLMReasoner` turns an `Observation` into a `Decision` via a single model call.
It depends only on a tiny `LLMClient` protocol (`complete(prompt) -> str`), so you
can point it at OpenAI, Bedrock, Vertex, a local vLLM server — anything. For tests
and demos, `DeterministicLLMClient` stands in for the model so runs stay offline and
reproducible:

```python
from loopengine import LLMReasoner, DeterministicLLMClient

reasoner = LLMReasoner(DeterministicLLMClient(extract))   # swap for a real client
# class MyClient:  def complete(self, prompt: str) -> str: return openai_call(prompt)
```

### 2 · Crew reasoner (CrewAI)

`CrewReasoner` promotes REASON into a **crew**: an Extractor proposes a value and an
Auditor validates and scores it. The crew still returns exactly one `Decision`, so
the loop is unchanged. It runs a deterministic **local crew** by default, and shows
the real **CrewAI** API when you opt in:

```python
from loopengine import CrewReasoner

reasoner = CrewReasoner(extract, use_crewai=False)   # deterministic local crew
# reasoner = CrewReasoner(extract, use_crewai=True)  # real crewai (needs an LLM)
```

The five determinism controls scale with the crew: **budgets** → `max_rpm` /
delegation depth, **stop conditions** → per-agent task completion, **guardrails** →
validate every inter-agent handoff, **idempotency** → content-addressed keys keep
retried tool calls safe, **replay** → log each agent's I/O. The warning label:
nondeterminism compounds with every agent, so pin seeds/temperatures and prefer
sequential handoffs for anything you must audit.

> **Contract, not magic:** rules today, an LLM tomorrow, a crew the day after —
> the controller, budgets, guardrails, idempotency, and replay never change.
> Let the model (or the crew) be creative; keep the loop deterministic.

## Project layout

```
loop-engine/
├── loopengine/
│   ├── state.py        # immutable phase I/O + LoopState (the control 'process variable')
│   ├── phases.py       # Sense/Reason/Act/Reflect protocols + Agent bundle
│   ├── reasoners.py    # LLMReasoner + CrewReasoner (CrewAI) behind one seam
│   ├── guardrails.py   # Budget + Guardrail policy primitives
│   ├── controller.py   # the deterministic engine (stop conditions live here)
│   └── replay.py       # tracing + deterministic replay/digest
├── examples/
│   ├── self_correcting_extractor.py   # the base loop (RuleReasoner)
│   └── reasoner_swap.py               # Rule vs LLM vs Crew on the same loop
└── tests/
    ├── test_controller.py
    └── test_reasoners.py
```

## Optional: real CrewAI

The core engine is dependency-free. To exercise the real CrewAI path
(`CrewReasoner(extract, use_crewai=True)`) install the extra and configure an LLM:

```bash
pip install crewai   # requires a configured model/provider to run kickoff()
```

Without it, `CrewReasoner` automatically uses its deterministic local crew, so the
example and tests run anywhere.

## License

MIT — see [LICENSE](LICENSE).
