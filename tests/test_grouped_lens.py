"""Tests for GroupedLens (Plan 2, E2 — grouped-lens task).

Design Decision D1: lenses read all_events(), never a side-car;
every lens recomputes from the totally-ordered log on each project().

E2 proof: GroupedLens produces the IDENTICAL payload bytes as RelationalLens,
grouped into a structurally different (inverted/grouped) shape.
"""

from __future__ import annotations

import pytest

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses import Lens
from amplifier_data.lenses.grouped import GroupedLens, QueryResult
from amplifier_data.lenses.relational import RelationalLens
from amplifier_data.models import (
    CellWriteEvent,
    InterpreterOutputEvent,
    RelationshipEvent,
    ToolResultEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interp_event(payload: bytes = b"out") -> InterpreterOutputEvent:
    """Construct a minimal InterpreterOutputEvent for testing."""
    cell_ref = InterpreterOutputEvent.derive_ref("model-1", "ih-1", (), payload)
    return InterpreterOutputEvent(
        cell_ref=cell_ref,
        model="model-1",
        input_hash="ih-1",
        tool_results_consumed=(),
        output=payload,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_grouped_lens_satisfies_lens_protocol() -> None:
    """GroupedLens satisfies the Lens Protocol (runtime_checkable)."""
    assert isinstance(GroupedLens(), Lens)


def test_grouped_lens_name_is_grouped() -> None:
    assert GroupedLens().name == "grouped"


# ---------------------------------------------------------------------------
# project() — pure-fold contract
# ---------------------------------------------------------------------------


def test_project_empty_kernel_returns_empty_dict() -> None:
    """project() on an empty kernel yields an empty grouped view."""
    view = GroupedLens().project(InMemoryKernel())
    assert view == {}


def test_project_returns_dict_not_list() -> None:
    """project() returns a dict (grouped shape), not a list (flat shape)."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"hello")])
    view = GroupedLens().project(kernel)
    assert isinstance(view, dict)


def test_project_cell_write_event_appears_in_group() -> None:
    """CellWriteEvent payload appears in the grouped view."""
    kernel = InMemoryKernel()
    ev = CellWriteEvent(payload=b"hello")
    kernel.append_batch([ev])

    view = GroupedLens().project(kernel)

    # The view must be non-empty and contain the payload somewhere
    all_payloads: list[bytes] = []
    for entries in view.values():
        for entry in entries:
            all_payloads.append(entry["payload"])

    assert b"hello" in all_payloads


def test_project_payload_bytes_are_verbatim_not_reshaped() -> None:
    """Payload bytes pass through unchanged — no summarising or reshaping."""
    raw = b"\x00\x01\x02\xff binary data \xfe"
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=raw)])

    view = GroupedLens().project(kernel)

    all_payloads: list[bytes] = []
    for entries in view.values():
        for entry in entries:
            all_payloads.append(entry["payload"])

    assert raw in all_payloads


def test_project_interpreter_output_event_appears_in_group() -> None:
    """InterpreterOutputEvent payload appears in the grouped view."""
    kernel = InMemoryKernel()
    ev = _make_interp_event(b"output bytes")
    kernel.append_batch([ev])

    view = GroupedLens().project(kernel)

    all_payloads: list[bytes] = []
    for entries in view.values():
        for entry in entries:
            all_payloads.append(entry["payload"])

    assert b"output bytes" in all_payloads


def test_project_tool_result_event_not_included() -> None:
    """ToolResultEvent is not cell-defining — must not appear in the view."""
    kernel = InMemoryKernel()
    cell_ev = CellWriteEvent(payload=b"cell")
    tool_ev = ToolResultEvent(output=b"tool output")
    kernel.append_batch([cell_ev, tool_ev])

    view = GroupedLens().project(kernel)

    all_payloads: list[bytes] = []
    for entries in view.values():
        for entry in entries:
            all_payloads.append(entry["payload"])

    # b"cell" is present, b"tool output" is NOT
    assert b"cell" in all_payloads
    assert b"tool output" not in all_payloads


def test_project_relationship_event_not_included() -> None:
    """RelationshipEvent is not cell-defining — must not appear in the view."""
    kernel = InMemoryKernel()
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    kernel.append_batch([a, b])
    kernel.append_batch(
        [RelationshipEvent(from_ref=a.cell_ref(), to_ref=b.cell_ref(), type="links_to")]
    )

    view = GroupedLens().project(kernel)

    total_entries = sum(len(entries) for entries in view.values())
    assert total_entries == 2  # only the two cell writes


def test_project_groups_are_deterministic() -> None:
    """The same kernel state always produces the same grouped view."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x"), CellWriteEvent(payload=b"y")])

    view1 = GroupedLens().project(kernel)
    view2 = GroupedLens().project(kernel)

    assert view1 == view2


def test_project_structurally_different_from_relational() -> None:
    """GroupedLens view is a dict — RelationalLens view is a list (different shapes)."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"data")])

    grouped_view = GroupedLens().project(kernel)
    relational_view = RelationalLens().project(kernel)

    assert isinstance(grouped_view, dict)
    assert isinstance(relational_view, list)
    assert type(grouped_view) is not type(relational_view)


def test_project_same_payload_bytes_as_relational() -> None:
    """E2 proof: GroupedLens contains the IDENTICAL payload bytes as RelationalLens."""
    kernel = InMemoryKernel()
    payloads = [b"alpha", b"beta", b"gamma"]
    kernel.append_batch([CellWriteEvent(payload=p) for p in payloads])

    relational_rows = RelationalLens().project(kernel)
    relational_payloads = sorted(r["payload"] for r in relational_rows)

    grouped_view = GroupedLens().project(kernel)
    grouped_payloads: list[bytes] = []
    for entries in grouped_view.values():
        for entry in entries:
            grouped_payloads.append(entry["payload"])
    grouped_payloads_sorted = sorted(grouped_payloads)

    assert grouped_payloads_sorted == relational_payloads


def test_project_same_payloads_mixed_event_types() -> None:
    """E2 proof holds for mixed CellWriteEvent and InterpreterOutputEvent."""
    kernel = InMemoryKernel()
    cell_payload = b"cell-payload"
    interp_payload = b"interp-payload"
    kernel.append_batch([CellWriteEvent(payload=cell_payload)])
    kernel.append_batch([_make_interp_event(interp_payload)])

    relational_rows = RelationalLens().project(kernel)
    relational_payloads = sorted(r["payload"] for r in relational_rows)

    grouped_view = GroupedLens().project(kernel)
    grouped_payloads: list[bytes] = []
    for entries in grouped_view.values():
        for entry in entries:
            grouped_payloads.append(entry["payload"])
    grouped_payloads_sorted = sorted(grouped_payloads)

    assert grouped_payloads_sorted == relational_payloads


def test_project_is_pure_fold_mutating_result_does_not_poison_next_call() -> None:
    """Mutating the returned dict does not affect the next project() call."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])
    lens = GroupedLens()

    view1 = lens.project(kernel)
    view1.clear()  # mutate the returned view

    view2 = lens.project(kernel)
    total = sum(len(entries) for entries in view2.values())
    assert total == 1  # recomputed — not poisoned


def test_project_reflects_newly_appended_events() -> None:
    """project() recomputes on every call — events appended after the first call appear."""
    kernel = InMemoryKernel()
    lens = GroupedLens()

    before = lens.project(kernel)
    assert before == {}

    kernel.append_batch([CellWriteEvent(payload=b"new")])
    after = lens.project(kernel)
    total = sum(len(entries) for entries in after.values())
    assert total == 1


def test_project_two_instances_are_independent() -> None:
    """Two GroupedLens instances over the same kernel produce independent views."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"shared")])

    lens_a = GroupedLens()
    lens_b = GroupedLens()

    view_a = lens_a.project(kernel)
    view_b = lens_b.project(kernel)
    assert view_a == view_b

    view_a.clear()  # mutate a's view
    view_b2 = lens_b.project(kernel)
    total = sum(len(entries) for entries in view_b2.values())
    assert total == 1  # b unaffected


def test_project_entries_preserve_seq_pos() -> None:
    """Each grouped entry includes the seq_pos from the kernel log."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"first"), CellWriteEvent(payload=b"second")])

    view = GroupedLens().project(kernel)

    seq_positions: list[int] = []
    for entries in view.values():
        for entry in entries:
            seq_positions.append(entry["seq_pos"])

    assert 0 in seq_positions
    assert 1 in seq_positions


# ---------------------------------------------------------------------------
# Grouping key — structural validation
# ---------------------------------------------------------------------------


def test_project_groups_cell_write_and_interpreter_output_separately() -> None:
    """CellWriteEvent and InterpreterOutputEvent are in different groups."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"cell")])
    kernel.append_batch([_make_interp_event(b"interp")])

    view = GroupedLens().project(kernel)

    # Collect which groups contain which payloads
    group_of_cell: str | None = None
    group_of_interp: str | None = None
    for key, entries in view.items():
        for entry in entries:
            if entry["payload"] == b"cell":
                group_of_cell = key
            if entry["payload"] == b"interp":
                group_of_interp = key

    # Both payloads must be in the view
    assert group_of_cell is not None, "cell payload not found in any group"
    assert group_of_interp is not None, "interp payload not found in any group"
    # They must be in DIFFERENT groups (structurally different grouping)
    assert group_of_cell != group_of_interp


# ---------------------------------------------------------------------------
# query() — typed surface
# ---------------------------------------------------------------------------


def test_query_no_args_requires_kernel() -> None:
    """query() without 'kernel' raises TypeError."""
    with pytest.raises(TypeError):
        GroupedLens().query()


def test_query_returns_query_result_shaped_object() -> None:
    """query() returns an object with success, output, error fields."""
    result = GroupedLens().query(kernel=InMemoryKernel())
    assert hasattr(result, "success")
    assert hasattr(result, "output")
    assert hasattr(result, "error")


def test_query_empty_kernel_returns_success_empty_output() -> None:
    """query() on an empty kernel returns success with empty output."""
    result = GroupedLens().query(kernel=InMemoryKernel())
    assert isinstance(result, QueryResult)
    assert result.success is True
    assert result.output == {}
    assert result.error is None


def test_query_no_filter_returns_all_groups() -> None:
    """query() with no filter returns all groups."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a"), CellWriteEvent(payload=b"b")])
    result = GroupedLens().query(kernel=kernel)
    assert result.success is True
    total = sum(len(entries) for entries in result.output.values())
    assert total == 2


def test_query_filter_by_group_key_returns_matching_group() -> None:
    """query(group=...) returns only the matching group's entries."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"cell")])
    kernel.append_batch([_make_interp_event(b"interp")])

    # Find the group key for the cell write
    full_view = GroupedLens().project(kernel)
    cell_write_group: str | None = None
    for key, entries in full_view.items():
        for entry in entries:
            if entry["payload"] == b"cell":
                cell_write_group = key
                break
    assert cell_write_group is not None

    result = GroupedLens().query(kernel=kernel, group=cell_write_group)
    assert result.success is True
    # Only entries from the matched group
    assert set(result.output.keys()) == {cell_write_group}
    all_payloads = [e["payload"] for e in result.output[cell_write_group]]
    assert b"cell" in all_payloads
    assert b"interp" not in all_payloads


def test_query_unknown_group_key_raises_key_error() -> None:
    """query(group=<unknown>) raises KeyError — never a silent empty."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])

    with pytest.raises(KeyError):
        GroupedLens().query(kernel=kernel, group="nonexistent_group_key")


def test_query_unknown_group_key_raises_even_with_empty_kernel() -> None:
    """Raising on unknown group key is consistent even when kernel is empty."""
    with pytest.raises(KeyError):
        GroupedLens().query(kernel=InMemoryKernel(), group="bogus")


def test_query_result_is_tool_result_shaped() -> None:
    """QueryResult has success, output, error — the ToolResult shape."""
    result = GroupedLens().query(kernel=InMemoryKernel())
    assert hasattr(result, "success")
    assert hasattr(result, "output")
    assert hasattr(result, "error")
    assert isinstance(result, QueryResult)
