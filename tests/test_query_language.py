"""Tests for the unified composable query layer (Plan 3g)."""

from __future__ import annotations

import json

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.query import Query, run_query


def test_scan_relational_all_and_scoped() -> None:
    store = AmplifierStore()
    a = store.write_cell(b"alpha")
    b = store.write_cell(b"beta")
    store.scope(a, "S")

    all_rows = store.query([{"op": "scan", "lens": "relational"}])
    assert {r["cell_ref"] for r in all_rows.output} == {a, b}

    scoped = store.query([{"op": "scan", "lens": "relational", "scope": "S"}])
    assert {r["cell_ref"] for r in scoped.output} == {a}


def test_filter_cmps() -> None:
    store = AmplifierStore()
    hello = store.write_cell(b"hello world")
    store.write_cell(b"goodbye")

    res = Query.scan("relational").filter("payload", "contains", "hello").execute(store)
    assert {r["cell_ref"] for r in res.output} == {hello}

    eq = Query.scan("relational").filter("cell_ref", "eq", hello).execute(store)
    assert {r["cell_ref"] for r in eq.output} == {hello}

    ne = Query.scan("relational").filter("cell_ref", "ne", hello).execute(store)
    assert hello not in {r["cell_ref"] for r in ne.output}


def test_traverse_after_scan() -> None:
    store = AmplifierStore()
    root = store.write_cell(b"root")
    child = store.write_cell(b"child")
    gc = store.write_cell(b"gc")
    store.relate(root, child, "fork")
    store.relate(child, gc, "fork")

    res = (
        Query.scan("relational")
        .filter("cell_ref", "eq", root)
        .traverse(max_hops=2, rel_type="fork")
        .execute(store)
    )
    assert {r["cell_ref"] for r in res.output} == {child, gc}


def test_facts_scan_and_filter() -> None:
    store = AmplifierStore()
    store.assert_fact("alice", "works_on", "acme")
    store.assert_fact("bob", "likes", "tea")

    res = Query.scan("facts").filter("predicate", "eq", "works_on").execute(store)
    assert [r["subject"] for r in res.output] == ["alice"]

    store.invalidate_fact("alice", "works_on", "acme")
    triples = {
        (r["subject"], r["predicate"], r["object"])
        for r in Query.scan("facts").execute(store).output
    }
    assert ("alice", "works_on", "acme") not in triples


def test_project_order_limit() -> None:
    store = AmplifierStore()
    for p in (b"c", b"a", b"b"):
        store.write_cell(p)

    res = (
        Query.scan("relational").project(["cell_ref"]).order_by("cell_ref").limit(2).execute(store)
    )
    assert all(set(r.keys()) == {"cell_ref"} for r in res.output)
    assert len(res.output) == 2
    refs = [r["cell_ref"] for r in res.output]
    assert refs == sorted(refs)

    desc = (
        Query.scan("relational")
        .project(["cell_ref"])
        .order_by("cell_ref", desc=True)
        .execute(store)
    )
    drefs = [r["cell_ref"] for r in desc.output]
    assert drefs == sorted(drefs, reverse=True)


def test_aggregate_count() -> None:
    store = AmplifierStore()
    store.write_cell(b"a")
    store.write_cell(b"b")
    res = Query.scan("relational").aggregate(["event_type"], "count").execute(store)
    assert sum(r["count"] for r in res.output) == 2


def test_ir_roundtrip_and_builder_parity() -> None:
    store = AmplifierStore()
    store.write_cell(b"x marks")

    q = (
        Query.scan("relational", scope="S")
        .filter("payload", "contains", "x")
        .project(["cell_ref"])
        .limit(5)
    )
    ir = q.to_ir()
    assert json.loads(json.dumps(ir)) == ir  # pure JSON data

    hand = [
        {"op": "scan", "lens": "relational", "scope": "S"},
        {"op": "filter", "field": "payload", "cmp": "contains", "value": "x"},
        {"op": "project", "fields": ["cell_ref"]},
        {"op": "limit", "n": 5},
    ]
    assert ir == hand
    assert Query.from_ir(ir).execute(store).output == run_query(store, hand).output


def test_determinism() -> None:
    store = AmplifierStore()
    store.write_cell(b"a")
    ir = [{"op": "scan", "lens": "relational"}]
    assert store.query(ir).output == store.query(ir).output


def test_query_appends_no_access_events() -> None:
    store = AmplifierStore()
    store.write_cell(b"a")
    store.assert_fact("x", "p", "y")
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    before = len(kernel.all_events())
    store.query([{"op": "scan", "lens": "relational"}])
    Query.scan("facts").execute(store)
    assert len(kernel.all_events()) == before


def test_data_errors_return_failure_not_raise() -> None:
    store = AmplifierStore()
    assert run_query(store, []).success is False
    assert (
        run_query(store, [{"op": "filter", "field": "x", "cmp": "eq", "value": 1}]).success is False
    )
    bad = run_query(store, [{"op": "scan", "lens": "relational"}, {"op": "bogus"}])
    assert bad.success is False and bad.error is not None
