"""Worked story: the unified composable query layer (Plan 3g).

Shows the two faces of one query:
  * a hand-written **QueryIR** (pure JSON data — the agentic surface), and
  * the equivalent **Query** fluent builder (the human surface),
producing identical results over a small delegation tree. Then a group-by/count
aggregate over the same store.

Run:  python examples/query_language.py   (exits 0 on success)
"""

from __future__ import annotations

import json

from amplifier_data import AmplifierStore
from amplifier_data.query import Query, run_query


def main() -> None:
    print("Unified composable query layer\n")

    store = AmplifierStore()
    root = store.write_cell(b"root session")
    child = store.write_cell(b"child A")
    grandchild = store.write_cell(b"grandchild")
    store.relate(root, child, "fork")
    store.relate(child, grandchild, "fork")
    store.scope(root, "wing:demo")

    # (a) Agentic surface: a query is pure JSON data an agent can emit.
    ir = [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "cell_ref", "cmp": "eq", "value": root},
        {"op": "traverse", "max_hops": 2, "rel_type": "fork", "from_field": "cell_ref"},
        {"op": "project", "fields": ["cell_ref"]},
        {"op": "order_by", "field": "cell_ref", "desc": False},
    ]
    assert json.loads(json.dumps(ir)) == ir  # genuinely JSON-serializable
    from_ir_result = run_query(store, ir)

    # (b) Human surface: the fluent builder compiles to the SAME IR.
    q = (
        Query.scan("relational")
        .filter("cell_ref", "eq", root)
        .traverse(max_hops=2, rel_type="fork")
        .project(["cell_ref"])
        .order_by("cell_ref")
    )
    assert q.to_ir() == ir
    builder_result = q.execute(store)

    reached = {r["cell_ref"] for r in builder_result.output}
    print("  scan -> filter(root) -> traverse(2, fork) -> project -> order_by")
    print(f"  reachable from root: {len(reached)} cells")
    assert reached == {child, grandchild}
    assert builder_result.output == from_ir_result.output
    print("  [\u2713] hand-written IR and fluent builder produce identical results\n")

    # Aggregate: group-by + count over the whole store.
    agg = Query.scan("relational").aggregate(["event_type"], "count").execute(store)
    total = sum(r["count"] for r in agg.output)
    print(f"  aggregate(event_type, count): {agg.output}")
    assert total == 3
    print("  [\u2713] group-by/count aggregate correct\n")

    print(
        "SUCCESS: one query, two faces — JSON QueryIR (agentic) and a fluent "
        "builder (human) — composing scan/filter/traverse/project/order_by/aggregate."
    )


if __name__ == "__main__":
    main()
