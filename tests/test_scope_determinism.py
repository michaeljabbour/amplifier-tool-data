"""Tests for ScopeIndex.canonical_bytes() — byte-stable determinism gate.

Three tests:
1. test_canonical_bytes_empty_is_empty
2. test_canonical_bytes_deterministic_across_instances_and_calls
3. test_canonical_bytes_changes_with_new_edge
"""

from __future__ import annotations

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO, fold_scope
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


def test_canonical_bytes_empty_is_empty() -> None:
    """canonical_bytes() on an empty-kernel fold returns b''.

    An empty ScopeIndex (no membership edges) must produce zero bytes.
    """
    kernel = InMemoryKernel()
    index = fold_scope(kernel)
    assert index.canonical_bytes() == b""


def test_canonical_bytes_deterministic_across_instances_and_calls() -> None:
    """Edges inserted in non-sorted order produce identical bytes across two folds.

    Proves canonical sorting: two independent ScopeIndex instances built from
    kernels with edges appended in reversed order must produce byte-identical
    output.  Repeated calls on the same instance must also return the same bytes.
    """
    # Edges in one insertion order
    kernel_a = InMemoryKernel()
    _append_scope_edge(kernel_a, "zzz_cell", "mmm_scope")
    _append_scope_edge(kernel_a, "aaa_cell", "zzz_scope")
    _append_scope_edge(kernel_a, "aaa_cell", "aaa_scope")

    # Same edges but appended in reverse order
    kernel_b = InMemoryKernel()
    _append_scope_edge(kernel_b, "aaa_cell", "aaa_scope")
    _append_scope_edge(kernel_b, "aaa_cell", "zzz_scope")
    _append_scope_edge(kernel_b, "zzz_cell", "mmm_scope")

    index_a = fold_scope(kernel_a)
    index_b = fold_scope(kernel_b)

    bytes_a1 = index_a.canonical_bytes()
    bytes_a2 = index_a.canonical_bytes()  # repeated call on same instance
    bytes_b1 = index_b.canonical_bytes()

    # Repeated calls on same instance are identical
    assert bytes_a1 == bytes_a2

    # Two independent instances with same logical content are byte-identical
    assert bytes_a1 == bytes_b1

    # Verify the expected canonical form explicitly:
    # sorted cells: ["aaa_cell", "zzz_cell"]
    # aaa_cell: sorted scopes ["aaa_scope", "zzz_scope"] -> b"aaa_cell:aaa_scope:zzz_scope\n"
    # zzz_cell: sorted scopes ["mmm_scope"]              -> b"zzz_cell:mmm_scope\n"
    expected = b"aaa_cell:aaa_scope:zzz_scope\nzzz_cell:mmm_scope\n"
    assert bytes_a1 == expected


def test_canonical_bytes_changes_with_new_edge() -> None:
    """Adding a second scope edge changes canonical_bytes().

    A ScopeIndex with one edge must produce different bytes from one with
    two edges — verifies that canonical_bytes() captures every edge and
    changes whenever the index content changes.
    """
    kernel_one = InMemoryKernel()
    _append_scope_edge(kernel_one, "cell_x", "scope_alpha")
    index_one = fold_scope(kernel_one)
    bytes_one = index_one.canonical_bytes()

    kernel_two = InMemoryKernel()
    _append_scope_edge(kernel_two, "cell_x", "scope_alpha")
    _append_scope_edge(kernel_two, "cell_x", "scope_beta")
    index_two = fold_scope(kernel_two)
    bytes_two = index_two.canonical_bytes()

    assert bytes_one != bytes_two
