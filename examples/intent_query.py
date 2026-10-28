"""Worked story: intent-compiled queries (Plan 4a).

Shows the full arc: intent (str) -> injected IntentCompiler -> stored plan
(an ordinary Cell with provenance edges) -> deterministic execution -> and,
reflexively, plans queryable back out of the store with the query layer
itself ("you query your queries").

Run:  python examples/intent_query.py   (exits 0 on success)
"""

from __future__ import annotations

from amplifier_data import AmplifierStore
from amplifier_data.intent import (
    COMPILED_FROM,
    RESULT_OF,
    IntentNotUnderstood,
    RuleBasedCompiler,
)


def main() -> None:
    print("Intent-compiled queries\n")

    store = AmplifierStore()
    compiler = RuleBasedCompiler()

    # ------------------------------------------------------------------ #
    # 1. Seed a few note cells.                                          #
    # ------------------------------------------------------------------ #
    alpha_release = store.write_cell(b"alpha release shipped")
    store.write_cell(b"beta bug found")
    alpha_retro = store.write_cell(b"alpha retro notes")
    store.assert_fact(alpha_release, "relates_to", alpha_retro)
    print("  seeded 3 note cells + 1 KG fact\n")

    # ------------------------------------------------------------------ #
    # 2. Compile: intent (str) -> stored plan (a Cell), reviewable.      #
    # ------------------------------------------------------------------ #
    plan = store.compile_intent(
        'cells containing "alpha"', compiler, compiler_id=compiler.compiler_id
    )
    decoded_ir = store.get_cell(plan).payload  # the plan cell IS the canonical IR bytes
    print(f"  plan ref:     {plan[:16]}...")
    print(f"  plan payload: {decoded_ir.decode('utf-8')}")
    print("  [\u2713] plan stored as an ordinary, reviewable Cell\n")

    # ------------------------------------------------------------------ #
    # 3. Execute: deterministic, reusable.                               #
    # ------------------------------------------------------------------ #
    run = store.execute_plan(plan)
    assert run.result.success is True
    matched_refs = {r["cell_ref"] for r in run.result.output}
    # Reflexivity note: the intent cell (verbatim text 'cells containing "alpha"')
    # and the plan cell (whose canonical JSON literally contains '"value":"alpha"')
    # are themselves ordinary cells written by compile_intent, so a relational scan
    # for "alpha" finds them too -- correctly, not a bug. The two originally-seeded
    # matches are a subset of the result.
    assert {alpha_release, alpha_retro} <= matched_refs
    print(f"  execute_plan(plan) matched {len(matched_refs)} cells (includes plan provenance)")

    # Reusable: executing the SAME plan ref again yields identical rows.
    run_again = store.execute_plan(plan)
    assert run_again.result.output == run.result.output
    print("  [\u2713] re-executing the same plan ref yields identical rows (reusable)\n")

    # ------------------------------------------------------------------ #
    # 4. Reflexive: a stored query, found by executing a stored query.   #
    # ------------------------------------------------------------------ #
    plans_plan = store.compile_intent("show plans", compiler)
    plans_run = store.execute_plan(plans_plan)
    assert plans_run.result.success is True
    found_subjects = {r["subject"] for r in plans_run.result.output}
    assert plan in found_subjects
    print(f"  'show plans' found {len(found_subjects)} compiled plan(s), including the first")

    # Same thing, in raw QueryIR form -- no store.query_plans() convenience needed.
    raw = store.query([{"op": "scan", "lens": "facts", "predicate": COMPILED_FROM}])
    assert plan in {r["subject"] for r in raw.output}
    print("  [\u2713] a query about queries, executed WITH a query (you query your queries)\n")

    # ------------------------------------------------------------------ #
    # 5. Result cell: provenance attached, opt-in.                       #
    # ------------------------------------------------------------------ #
    recorded = store.execute_plan(plan, record_result=True)
    assert recorded.result_ref is not None
    linked = store.query_facts(subject=recorded.result_ref, predicate=RESULT_OF)
    assert len(linked.output) == 1 and linked.output[0].object == plan
    print(f"  result cell:  {recorded.result_ref[:16]}...")
    print("  [\u2713] recorded result cell carries @plan:result_of provenance back to its plan\n")

    # ------------------------------------------------------------------ #
    # 6. Loud failure: invalid intent is rejected; the log is untouched. #
    # ------------------------------------------------------------------ #
    before = len(store.kernel.all_events())  # type: ignore[attr-defined]
    try:
        store.compile_intent("do something vague", compiler)
        raise AssertionError("expected IntentNotUnderstood")
    except IntentNotUnderstood:
        pass
    after = len(store.kernel.all_events())  # type: ignore[attr-defined]
    assert after == before
    print("  invalid intent rejected; log untouched")
    print("  [\u2713] honest stopping: no guessed default plan, no partial writes\n")

    print(
        "SUCCESS: intent -> injected compiler -> stored plan Cell -> deterministic "
        "execution -> queryable back out with the query layer itself."
    )


if __name__ == "__main__":
    main()
