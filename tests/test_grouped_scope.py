"""Tests for scope= keyword in GroupedLens.query (3d completion — F3/F6).

Mirrors test_relational_scope.py: scope filters by direct SCOPED_TO edges only
(F3, no traversal); an unknown scope yields empty output + success=True (F6).
"""

from __future__ import annotations

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO
from amplifier_data.lenses.grouped import GroupedLens
from amplifier_data.models import CellWriteEvent, RelationshipEvent


def _append_scope_edge(kernel: InMemoryKernel, cell_ref: str, scope_ref: str) -> None:
    kernel.append_batch([RelationshipEvent(type=SCOPED_TO, from_ref=cell_ref, to_ref=scope_ref)])


def _refs(output: dict[str, list[dict]]) -> set[str]:
    return {entry["cell_ref"] for entries in output.values() for entry in entries}


def test_grouped_scope_keeps_only_in_scope_entries() -> None:
    kernel = InMemoryKernel()
    inside = CellWriteEvent(payload=b"inside")
    outside = CellWriteEvent(payload=b"outside")
    kernel.append_batch([inside, outside])

    scope_ref = "scope_ref_grp_a"
    _append_scope_edge(kernel, inside.cell_ref(), scope_ref)

    result = GroupedLens().query(kernel=kernel, scope=scope_ref)

    assert result.success is True
    assert _refs(result.output) == {inside.cell_ref()}


def test_grouped_scope_unknown_is_empty_success() -> None:
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])

    result = GroupedLens().query(kernel=kernel, scope="nonexistent_grp_scope")

    assert result.success is True
    assert result.output == {}


def test_grouped_scope_composes_with_group_key() -> None:
    kernel = InMemoryKernel()
    keep = CellWriteEvent(payload=b"keep")
    drop = CellWriteEvent(payload=b"drop")
    kernel.append_batch([keep, drop])

    scope_ref = "scope_ref_grp_b"
    _append_scope_edge(kernel, keep.cell_ref(), scope_ref)

    result = GroupedLens().query(kernel=kernel, scope=scope_ref, group="cell_write")

    assert result.success is True
    assert _refs(result.output) == {keep.cell_ref()}


def test_grouped_scope_hierarchy_two_edges() -> None:
    """A cell scoped to both room and wing is returned by either scope."""
    kernel = InMemoryKernel()
    drawer = CellWriteEvent(payload=b"drawer")
    kernel.append_batch([drawer])

    room, wing = "room_ref", "wing_ref"
    _append_scope_edge(kernel, drawer.cell_ref(), room)
    _append_scope_edge(kernel, drawer.cell_ref(), wing)

    by_room = GroupedLens().query(kernel=kernel, scope=room)
    by_wing = GroupedLens().query(kernel=kernel, scope=wing)

    assert _refs(by_room.output) == {drawer.cell_ref()}
    assert _refs(by_wing.output) == {drawer.cell_ref()}


def test_grouped_no_scope_unchanged() -> None:
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a"), CellWriteEvent(payload=b"b")])

    result = GroupedLens().query(kernel=kernel)

    assert result.success is True
    assert len(result.output["cell_write"]) == 2


def test_grouped_scope_deterministic() -> None:
    kernel = InMemoryKernel()
    a = CellWriteEvent(payload=b"a")
    kernel.append_batch([a])
    _append_scope_edge(kernel, a.cell_ref(), "s")

    r1 = GroupedLens().query(kernel=kernel, scope="s")
    r2 = GroupedLens().query(kernel=kernel, scope="s")
    assert r1.output == r2.output
