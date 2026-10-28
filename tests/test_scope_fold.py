"""Unit tests for _scope.py — scope fold (fold_scope / ScopeIndex).

Design Decision F3 (direct edges only) is verified by
test_fold_records_only_direct_edges_no_traversal: a chain
cell -> scope_a -> scope_b must NOT put scope_b in cell's membership.

Five tests mirror the acceptance criteria from the task spec:
1. test_scoped_to_constant_value
2. test_fold_empty_kernel_is_empty
3. test_fold_single_direct_edge
4. test_fold_records_only_direct_edges_no_traversal
5. test_fold_cell_in_multiple_scopes
"""

from __future__ import annotations

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO, ScopeIndex, fold_scope
from amplifier_data.models import RelationshipEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_scope_edge(kernel: InMemoryKernel, cell_ref: str, scope_ref: str) -> None:
    """Append a SCOPED_TO RelationshipEvent to the kernel."""
    event = RelationshipEvent(type=SCOPED_TO, from_ref=cell_ref, to_ref=scope_ref)
    kernel.append_batch([event])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scoped_to_constant_value() -> None:
    """SCOPED_TO must equal the canonical string 'scoped_to'."""
    assert SCOPED_TO == "scoped_to"


def test_fold_empty_kernel_is_empty() -> None:
    """fold_scope on an empty kernel returns ScopeIndex with membership == {}."""
    kernel = InMemoryKernel()
    result = fold_scope(kernel)
    assert isinstance(result, ScopeIndex)
    assert result.membership == {}


def test_fold_single_direct_edge() -> None:
    """A single SCOPED_TO edge cell->scope yields {cell: frozenset({scope})}."""
    kernel = InMemoryKernel()
    cell = "cell_aaa"
    scope = "scope_bbb"
    _append_scope_edge(kernel, cell, scope)

    result = fold_scope(kernel)
    assert result.membership == {cell: frozenset({scope})}


def test_fold_records_only_direct_edges_no_traversal() -> None:
    """F3 — direct edges only: cell->scope_a, scope_a->scope_b.

    cell's membership must be frozenset({scope_a}).
    scope_b must NOT appear in cell's membership (no transitive closure).
    """
    kernel = InMemoryKernel()
    cell = "cell_001"
    scope_a = "scope_a"
    scope_b = "scope_b"

    # Direct edge: cell is in scope_a
    _append_scope_edge(kernel, cell, scope_a)
    # Scope-to-scope edge: scope_a is in scope_b (should NOT be followed)
    _append_scope_edge(kernel, scope_a, scope_b)

    result = fold_scope(kernel)

    # cell's membership is exactly {scope_a}
    assert result.membership.get(cell) == frozenset({scope_a})
    # scope_b must NOT appear in cell's membership — F3: no transitive closure
    assert scope_b not in result.membership.get(cell, frozenset())


def test_fold_cell_in_multiple_scopes() -> None:
    """A cell scoped to two scopes yields frozenset({s1, s2})."""
    kernel = InMemoryKernel()
    cell = "cell_multi"
    s1 = "scope_s1"
    s2 = "scope_s2"

    _append_scope_edge(kernel, cell, s1)
    _append_scope_edge(kernel, cell, s2)

    result = fold_scope(kernel)
    assert result.membership.get(cell) == frozenset({s1, s2})


# ---------------------------------------------------------------------------
# Task 2: ScopeIndex membership API — is_in_scope / cells_in_scope
# ---------------------------------------------------------------------------


def test_is_in_scope_true_and_false() -> None:
    """is_in_scope returns True for a direct SCOPED_TO edge, False for non-members.

    Verifies: is_in_scope(cell, scope) is True, is_in_scope(other, scope) is False.
    """
    kernel = InMemoryKernel()
    cell = "cell_alpha"
    other = "cell_beta"
    scope = "scope_xyz"

    _append_scope_edge(kernel, cell, scope)
    # 'other' has no edge at all

    index = fold_scope(kernel)

    assert index.is_in_scope(cell, scope) is True
    assert index.is_in_scope(other, scope) is False


def test_cells_in_scope_collects_members() -> None:
    """cells_in_scope collects all cells directly scoped to the given scope.

    Two cells scoped to the same scope must yield frozenset({c1, c2}).
    """
    kernel = InMemoryKernel()
    c1 = "cell_one"
    c2 = "cell_two"
    scope = "scope_shared"

    _append_scope_edge(kernel, c1, scope)
    _append_scope_edge(kernel, c2, scope)

    index = fold_scope(kernel)

    assert index.cells_in_scope(scope) == frozenset({c1, c2})


def test_unknown_scope_returns_empty_not_error() -> None:
    """F6 — unknown scope yields empty frozenset, never raises an error.

    cells_in_scope of an unknown scope_ref == frozenset().
    is_in_scope with an unknown cell_ref or unknown scope_ref is False.
    """
    kernel = InMemoryKernel()
    # No events appended — completely empty kernel
    index = fold_scope(kernel)

    assert index.cells_in_scope("nonexistent_scope") == frozenset()
    assert index.is_in_scope("nonexistent_cell", "nonexistent_scope") is False
    assert index.is_in_scope("nonexistent_cell", "nonexistent_scope") is False
