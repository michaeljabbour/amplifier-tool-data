"""AmplifierStore.add_to_scope must be a true alias for scope (parity with RemoteStore)."""

from __future__ import annotations

from amplifier_data import AmplifierStore


def test_add_to_scope_is_alias_for_scope() -> None:
    store = AmplifierStore()
    c = store.write_cell(b"content")

    store.add_to_scope(c, "wing:demo")

    # Same effect as scope(): a scoped_to edge that scope= queries can see.
    assert store.graph_neighbors(c, rel_type="scoped_to") == ["wing:demo"]
    refs = {r["cell_ref"] for r in store.query_relational(scope="wing:demo").output}
    assert c in refs


def test_add_to_scope_and_scope_are_the_same_callable() -> None:
    assert AmplifierStore.add_to_scope is AmplifierStore.scope
