"""Tests for the projection / visualization layer (Plan 3h)."""

from __future__ import annotations

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.projection import build_projection, to_dot, to_html


def _store() -> tuple[AmplifierStore, str, str, str]:
    store = AmplifierStore()
    root = store.write_cell(b"root")
    child = store.write_cell(b"child")
    other = store.write_cell(b"other")
    store.relate(root, child, "fork")
    store.scope(root, "wing:demo")
    store.assert_fact("alice", "works_on", "acme")
    store.add_embedding(child, [0.1, 0.2])  # embedding_of edge — must NOT appear as a generic edge
    return store, root, child, other


def test_build_projection_contents() -> None:
    store, root, child, _other = _store()
    proj = build_projection(store)

    refs = {r for r, _ in proj.nodes}
    assert root in refs and child in refs
    assert (root, child, "fork") in proj.edges
    assert (root, "wing:demo") in proj.scopes
    assert ("alice", "works_on", "acme") in proj.facts
    # embedding_of and scoped_to are not generic edges
    assert all(t not in ("embedding_of", "scoped_to") for _, _, t in proj.edges)


def test_projection_restricted_to_cell_refs() -> None:
    store, root, child, other = _store()
    proj = build_projection(store, cell_refs=[root, child])
    refs = {r for r, _ in proj.nodes}
    assert other not in refs
    assert (root, child, "fork") in proj.edges


def test_to_dot_deterministic_and_contains_edges() -> None:
    store, root, child, _other = _store()
    proj = build_projection(store)
    dot1 = to_dot(proj)
    dot2 = to_dot(build_projection(store))
    assert dot1 == dot2  # deterministic
    assert dot1.startswith("digraph amplifier_data {")
    assert f'"{root}" -> "{child}"' in dot1
    assert "fork" in dot1


def test_to_html_is_self_contained() -> None:
    store, root, _child, _other = _store()
    proj = build_projection(store)
    page = to_html(proj, title="demo")
    assert page.startswith("<!DOCTYPE html>")
    assert "<svg" in page and "</svg>" in page
    # self-contained: no external network references
    assert "http://" not in page and "https://" not in page
    assert "alice" in page  # fact rendered
    assert root[:8] in page


def test_store_sugar() -> None:
    store, root, child, _other = _store()
    assert store.to_dot().startswith("digraph")
    assert store.to_html().startswith("<!DOCTYPE html>")
    assert f'"{root}" -> "{child}"' in store.to_dot()


def test_projection_appends_no_access_events() -> None:
    store, _root, _child, _other = _store()
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    before = len(kernel.all_events())
    store.to_dot()
    store.to_html()
    assert len(kernel.all_events()) == before
