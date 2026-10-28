"""Tests for GraphLens traversal (Plan 3f-graph).

Covers bounded BFS (max_hops), edge-type filtering, cycle safety, direct
neighbors, determinism, and durable restart survival. Exercises the store
sugar (query_graph / graph_neighbors) which reads the kernel directly and
appends no AccessEvents (D4).
"""

from __future__ import annotations

from pathlib import Path

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses.graph import GraphLens


def _tree() -> tuple[AmplifierStore, str, str, str, str]:
    """root --fork--> child --fork--> grandchild ; plus an unrelated 'other'."""
    store = AmplifierStore()
    root = store.write_cell(b"root")
    child = store.write_cell(b"child")
    grandchild = store.write_cell(b"grandchild")
    other = store.write_cell(b"other")
    store.relate(root, child, "fork")
    store.relate(child, grandchild, "fork")
    return store, root, child, grandchild, other


def test_traverse_max_hops_2() -> None:
    store, root, child, grandchild, other = _tree()
    out = store.query_graph(root, max_hops=2).output
    assert out == sorted([child, grandchild])
    assert other not in out


def test_traverse_max_hops_1_direct_only() -> None:
    store, root, child, _grandchild, _other = _tree()
    assert store.query_graph(root, max_hops=1).output == [child]


def test_unrelated_node_never_reached() -> None:
    store, root, _child, _grandchild, other = _tree()
    assert other not in store.query_graph(root, max_hops=9).output


def test_type_filter_restricts_walk() -> None:
    store = AmplifierStore()
    a = store.write_cell(b"a")
    b = store.write_cell(b"b")
    c = store.write_cell(b"c")
    store.relate(a, b, "x")
    store.relate(a, c, "y")
    assert store.query_graph(a, max_hops=1, rel_type="x").output == [b]
    assert store.query_graph(a, max_hops=1, rel_type="y").output == [c]


def test_cycle_terminates() -> None:
    store = AmplifierStore()
    a = store.write_cell(b"a")
    b = store.write_cell(b"b")
    store.relate(a, b, "e")
    store.relate(b, a, "e")  # cycle a -> b -> a
    out = store.query_graph(a, max_hops=5).output  # must terminate
    assert out == sorted([a, b])  # a is reachable from itself via the cycle


def test_neighbors_direct_and_filtered() -> None:
    store = AmplifierStore()
    a = store.write_cell(b"a")
    b = store.write_cell(b"b")
    c = store.write_cell(b"c")
    store.relate(a, b, "x")
    store.relate(a, c, "y")
    assert set(store.graph_neighbors(a)) == {b, c}
    assert store.graph_neighbors(a, rel_type="x") == [b]
    assert store.graph_neighbors(c) == []  # no out-edges


def test_traverse_deterministic() -> None:
    store, root, _child, _gc, _other = _tree()
    assert store.query_graph(root, max_hops=3).output == store.query_graph(root, max_hops=3).output


def test_graph_query_reads_append_no_access_events() -> None:
    """query_graph / graph_neighbors must not grow the log (D4)."""
    store, root, _child, _gc, _other = _tree()
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    before = len(kernel.all_events())
    store.query_graph(root, max_hops=3)
    store.graph_neighbors(root)
    assert len(kernel.all_events()) == before


def test_durable_restart_survival(tmp_path: Path) -> None:
    log = tmp_path / "graph.log"
    store = AmplifierStore(path=str(log))
    root = store.write_cell(b"root")
    child = store.write_cell(b"child")
    gc = store.write_cell(b"gc")
    store.relate(root, child, "fork")
    store.relate(child, gc, "fork")
    before = store.query_graph(root, max_hops=2).output
    store.close()

    reopened = AmplifierStore(path=str(log))
    after = reopened.query_graph(root, max_hops=2).output
    reopened.close()
    assert after == before == sorted([child, gc])


def test_lens_level_traverse_matches_store() -> None:
    store, root, child, gc, _other = _tree()
    lens = GraphLens()
    assert lens.traverse(store.kernel, root, max_hops=2) == sorted([child, gc])
