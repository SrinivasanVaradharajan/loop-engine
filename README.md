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
| **Reason** | "What should I do next?" | `RuleReasoner.reason()` → `Decision` |
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

The suite proves the four properties that matter:

- `test_loop_converges` — the loop reaches a fully valid record.
- `test_determinism_same_trace_hash` — two runs produce a **byte-identical** trace.
- `test_budget_is_a_hard_stop` — autonomy is bounded by budget.
- `test_guardrail_blocks_unapproved_action` — policy halts the loop safely.

---

## Wiring in a real model

The `Reasoner` protocol is the only seam you change to go from rules to an LLM:

```python
class LLMReasoner:
    def reason(self, state, obs):
        resp = my_llm.complete(prompt=build_prompt(state, obs))
        action = parse(resp)
        return Decision(
            action=action.name,
            arguments=action.args,
            rationale=action.why,
            confidence=action.confidence,
            idempotency_key=f"{action.name}:{stable_hash(action.args)}",
        )
```

Everything else — budgets, guardrails, idempotency, replay — stays exactly the same.
That separation is the whole point: **let the model be creative; keep the loop deterministic.**

## Project layout

```
loop-engine/
├── loopengine/
│   ├── state.py        # immutable phase I/O + LoopState (the control 'process variable')
│   ├── phases.py       # Sense/Reason/Act/Reflect protocols + Agent bundle
│   ├── guardrails.py   # Budget + Guardrail policy primitives
│   ├── controller.py   # the deterministic engine (stop conditions live here)
│   └── replay.py       # tracing + deterministic replay/digest
├── examples/
│   └── self_correcting_extractor.py
└── tests/
    └── test_controller.py
```

## License

MIT — see [LICENSE](LICENSE).
