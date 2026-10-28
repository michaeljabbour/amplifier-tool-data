"""Tests for scope= keyword in IndexLens.query (3d completion — F3/F6).

IndexLens folds AccessEvents into a {cell_ref: count} table. scope= restricts
the returned counts to cells with a direct SCOPED_TO edge to the given scope
(F3, no traversal); an unknown scope yields empty output + success=True (F6).
The reflexive index fold itself is unchanged — scope is a read-time filter.
"""

from __future__ import annotations

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO
from amplifier_data.lenses.index import IndexLens
from amplifier_data.models import AccessEvent, CellWriteEvent, RelationshipEvent


def _scope_edge(kernel: InMemoryKernel, cell_ref: str, scope_ref: str) -> None:
    kernel.append_batch([RelationshipEvent(type=SCOPED_TO, from_ref=cell_ref, to_ref=scope_ref)])


def _build() -> tuple[InMemoryKernel, str, str]:
    """Two cells; cell A read twice, cell B read once; A scoped to 's'."""
    kernel = InMemoryKernel()
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    kernel.append_batch([a, b])
    a_ref, b_ref = a.cell_ref(), b.cell_ref()
    kernel.append_batch([AccessEvent(cell_ref=a_ref), AccessEvent(cell_ref=a_ref)])
    kernel.append_batch([AccessEvent(cell_ref=b_ref)])
    _scope_edge(kernel, a_ref, "s")
    return kernel, a_ref, b_ref


def test_index_scope_keeps_only_in_scope_counts() -> None:
    kernel, a_ref, b_ref = _build()

    result = IndexLens().query(kernel=kernel, scope="s")

    assert result.success is True
    assert result.output == {a_ref: 2}  # b_ref excluded — not in scope


def test_index_scope_unknown_is_empty_success() -> None:
    kernel, _a, _b = _build()

    result = IndexLens().query(kernel=kernel, scope="nonexistent_idx_scope")

    assert result.success is True
    assert result.output == {}


def test_index_no_scope_unchanged() -> None:
    kernel, a_ref, b_ref = _build()

    result = IndexLens().query(kernel=kernel)

    assert result.success is True
    assert result.output == {a_ref: 2, b_ref: 1}


def test_index_scope_hierarchy_two_edges() -> None:
    kernel, a_ref, _b = _build()
    _scope_edge(kernel, a_ref, "wing")  # A now in both 's' and 'wing'

    assert IndexLens().query(kernel=kernel, scope="s").output == {a_ref: 2}
    assert IndexLens().query(kernel=kernel, scope="wing").output == {a_ref: 2}


def test_index_scope_deterministic() -> None:
    kernel, _a, _b = _build()
    r1 = IndexLens().query(kernel=kernel, scope="s")
    r2 = IndexLens().query(kernel=kernel, scope="s")
    assert r1.output == r2.output
