"""Tests for scope= keyword in RelationalLens.query (Task 5 — F3/F6).

Design Decision F3: scope filter uses direct SCOPED_TO edges only (no traversal).
Design Decision F6: unknown scope yields empty output + success=True, never raises.
"""

from __future__ import annotations

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO
from amplifier_data.lenses.relational import RelationalLens
from amplifier_data.models import CellWriteEvent, RelationshipEvent

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


def test_scope_filter_keeps_only_in_scope_rows() -> None:
    """scope= keeps only rows whose cell_ref is directly SCOPED_TO the given scope.

    success must be True; refs must equal {inside_ref} (outside row excluded).
    """
    kernel = InMemoryKernel()
    inside_ev = CellWriteEvent(payload=b"inside")
    outside_ev = CellWriteEvent(payload=b"outside")
    kernel.append_batch([inside_ev, outside_ev])

    scope_ref = "scope_ref_aaa"
    _append_scope_edge(kernel, inside_ev.cell_ref(), scope_ref)
    # outside_ev has no SCOPED_TO edge — must be excluded

    lens = RelationalLens()
    result = lens.query(kernel=kernel, scope=scope_ref)

    assert result.success is True
    refs = {r["cell_ref"] for r in result.output}
    assert refs == {inside_ev.cell_ref()}


def test_scope_filter_unknown_scope_is_empty_success() -> None:
    """F6 — unknown scope yields empty output with success=True, never raises."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"some cell")])

    lens = RelationalLens()
    result = lens.query(kernel=kernel, scope="nonexistent_scope_xyz")

    assert result.success is True
    assert result.output == []


def test_scope_composes_with_user_filter() -> None:
    """scope= composes with filter=: scope narrows rows first, then filter applies.

    Two cells are inside scope with payloads b"keep me" and b"skip me".
    One cell outside scope also has payload b"keep me outside scope".

    After scope= two rows are candidates (keep_ev, skip_ev).
    After filter= (b"keep" in payload) only keep_ev survives.
    refs must equal {keep_ev.cell_ref()}.
    """
    kernel = InMemoryKernel()
    keep_ev = CellWriteEvent(payload=b"keep me")
    skip_ev = CellWriteEvent(payload=b"skip me")
    outside_ev = CellWriteEvent(payload=b"keep me outside scope")
    kernel.append_batch([keep_ev, skip_ev, outside_ev])

    scope_ref = "scope_ref_bbb"
    _append_scope_edge(kernel, keep_ev.cell_ref(), scope_ref)
    _append_scope_edge(kernel, skip_ev.cell_ref(), scope_ref)
    # outside_ev is NOT in scope — must be excluded even though filter would keep it

    lens = RelationalLens()
    result = lens.query(
        kernel=kernel,
        scope=scope_ref,
        filter=lambda r: b"keep" in r["payload"],
    )

    assert result.success is True
    refs = {r["cell_ref"] for r in result.output}
    assert refs == {keep_ev.cell_ref()}


def test_no_scope_kwarg_unchanged_behaviour() -> None:
    """Without scope=, query() returns all rows — unchanged behaviour (2 rows)."""
    kernel = InMemoryKernel()
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    kernel.append_batch([a, b])

    lens = RelationalLens()
    result = lens.query(kernel=kernel)

    assert result.success is True
    assert len(result.output) == 2
