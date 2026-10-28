"""Worked story: GraphLens delegation-tree traversal + durable restart (Plan 3f-graph).

Builds a small delegation tree of cells connected by typed ``fork`` edges in a
DURABLE store, traverses it (bounded BFS + direct neighbors), then closes and
reopens the store on the same path and re-traverses — proving the graph view is
a deterministic fold of the event log that survives a process restart (E1).

Run:  python examples/graph_traverse.py   (exits 0 on success)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from amplifier_data import AmplifierStore


def main() -> None:
    print("GraphLens traversal + durable restart\n")

    with tempfile.TemporaryDirectory() as tmp:
        log = str(Path(tmp) / "tree.log")

        # ── Session 1: build a delegation tree, traverse it ──────────────────
        store = AmplifierStore(path=log)
        root = store.write_cell(b"root session")
        child_a = store.write_cell(b"child A")
        child_b = store.write_cell(b"child B")
        grandchild = store.write_cell(b"grandchild")
        store.relate(root, child_a, "fork")
        store.relate(root, child_b, "fork")
        store.relate(child_a, grandchild, "fork")

        direct = store.query_graph(root, max_hops=1).output
        whole = store.query_graph(root, max_hops=2).output
        neighbors = store.graph_neighbors(root)

        print(f"  direct children (max_hops=1): {len(direct)}")
        print(f"  whole subtree   (max_hops=2): {len(whole)}")
        assert direct == sorted([child_a, child_b])
        assert whole == sorted([child_a, child_b, grandchild])
        assert set(neighbors) == {child_a, child_b}
        print("  [\u2713] bounded BFS + direct neighbors correct")

        store.close()
        print("  closed store (flush + release file handle)\n")

        # ── Session 2: reopen on the same path, re-traverse ──────────────────
        reopened = AmplifierStore(path=log)
        whole_after = reopened.query_graph(root, max_hops=2).output
        reopened.close()

        assert whole_after == whole
        print(f"  before restart: {len(whole)} reachable")
        print(f"  after  restart: {len(whole_after)} reachable")
        print("  [\u2713] traversal identical across a durable restart (E1)\n")

    print(
        "SUCCESS: graph traversal — bounded BFS, direct neighbors, and "
        "byte-identical regeneration across a durable restart."
    )


if __name__ == "__main__":
    main()
