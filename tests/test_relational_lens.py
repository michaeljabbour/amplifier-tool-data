"""Tests for RelationalLens (Plan 2, E2 — relational-lens task).

Design Decision D1: lenses read all_events(), never a side-car;
every lens recomputes from the totally-ordered log on each project().
"""

from __future__ import annotations

import pytest

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses import Lens
from amplifier_data.lenses.relational import QueryResult, RelationalLens
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


def test_relational_lens_satisfies_lens_protocol() -> None:
    """RelationalLens satisfies the Lens Protocol (runtime_checkable)."""
    assert isinstance(RelationalLens(), Lens)


def test_relational_lens_name_is_relational() -> None:
    assert RelationalLens().name == "relational"


# ---------------------------------------------------------------------------
# project() — pure-fold contract
# ---------------------------------------------------------------------------


def test_project_empty_kernel_returns_empty_list() -> None:
    """project() on an empty kernel yields no rows."""
    rows = RelationalLens().project(InMemoryKernel())
    assert rows == []


def test_project_cell_write_event_becomes_one_row() -> None:
    """Each CellWriteEvent produces exactly one row."""
    kernel = InMemoryKernel()
    ev = CellWriteEvent(payload=b"hello")
    kernel.append_batch([ev])

    rows = RelationalLens().project(kernel)

    assert len(rows) == 1
    row = rows[0]
    assert row["payload"] == b"hello"
    assert row["cell_ref"] == ev.cell_ref()
    assert row["event_type"] == "cell_write"
    assert row["seq_pos"] == 0
    assert row["interpreters"] == ()
    assert row["model"] is None
    assert row["input_hash"] is None


def test_project_payload_is_verbatim_not_reshaped() -> None:
    """Payload bytes pass through unchanged — no summarising or reshaping."""
    raw = b"\x00\x01\x02\xff binary data \xfe"
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=raw)])

    rows = RelationalLens().project(kernel)
    assert rows[0]["payload"] == raw


def test_project_interpreter_output_event_becomes_one_row() -> None:
    """Each InterpreterOutputEvent produces exactly one row."""
    kernel = InMemoryKernel()
    ev = _make_interp_event(b"output bytes")
    kernel.append_batch([ev])

    rows = RelationalLens().project(kernel)

    assert len(rows) == 1
    row = rows[0]
    assert row["payload"] == b"output bytes"
    assert row["cell_ref"] == ev.cell_ref
    assert row["event_type"] == "interpreter_output"
    assert row["model"] == "model-1"
    assert row["input_hash"] == "ih-1"


def test_project_tool_result_event_not_included() -> None:
    """ToolResultEvent is not a cell-defining event — it must not appear in rows."""
    kernel = InMemoryKernel()
    cell_ev = CellWriteEvent(payload=b"cell")
    tool_ev = ToolResultEvent(output=b"tool output")
    kernel.append_batch([cell_ev, tool_ev])

    rows = RelationalLens().project(kernel)
    assert len(rows) == 1  # only the cell; not the tool result
    assert rows[0]["event_type"] == "cell_write"


def test_project_relationship_event_not_included() -> None:
    """RelationshipEvent is not a cell-defining event — it must not appear in rows."""
    kernel = InMemoryKernel()
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    kernel.append_batch([a, b])
    kernel.append_batch(
        [RelationshipEvent(from_ref=a.cell_ref(), to_ref=b.cell_ref(), type="links_to")]
    )

    rows = RelationalLens().project(kernel)
    assert len(rows) == 2  # only the two cells; not the edge


def test_project_multiple_events_correct_seq_pos() -> None:
    """seq_pos in each row reflects the event's position in the kernel log."""
    kernel = InMemoryKernel()
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    kernel.append_batch([a, b])

    rows = RelationalLens().project(kernel)

    assert len(rows) == 2
    assert rows[0]["seq_pos"] == 0
    assert rows[1]["seq_pos"] == 1


def test_project_is_pure_fold_mutating_result_does_not_poison_next_call() -> None:
    """Mutating the returned list does not affect the next project() call."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])
    lens = RelationalLens()

    rows1 = lens.project(kernel)
    rows1.append(
        {
            "seq_pos": 999,
            "cell_ref": "fake",
            "payload": b"bogus",
            "event_type": "fake",
            "interpreters": (),
            "model": None,
            "input_hash": None,
        }
    )

    rows2 = lens.project(kernel)
    assert len(rows2) == 1  # recomputed — not poisoned


def test_project_reflects_newly_appended_events() -> None:
    """project() recomputes on every call — events appended after the first call appear."""
    kernel = InMemoryKernel()
    lens = RelationalLens()

    before = lens.project(kernel)
    assert before == []

    kernel.append_batch([CellWriteEvent(payload=b"new")])
    after = lens.project(kernel)
    assert len(after) == 1


def test_project_two_instances_are_independent() -> None:
    """Two RelationalLens instances over the same kernel produce independent views."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"shared")])

    lens_a = RelationalLens()
    lens_b = RelationalLens()

    rows_a = lens_a.project(kernel)
    rows_b = lens_b.project(kernel)
    assert rows_a == rows_b

    rows_a.clear()  # mutate a's view
    rows_b2 = lens_b.project(kernel)
    assert len(rows_b2) == 1  # b unaffected


# ---------------------------------------------------------------------------
# query() — typed surface
# ---------------------------------------------------------------------------


def test_query_no_filter_no_project_returns_all_rows() -> None:
    """query() with no filter and no project returns all rows with all columns."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a"), CellWriteEvent(payload=b"b")])
    lens = RelationalLens()

    result = lens.query(kernel=kernel)

    assert isinstance(result, QueryResult)
    assert result.success is True
    assert result.error is None
    assert len(result.output) == 2


def test_query_result_is_tool_result_shaped() -> None:
    """QueryResult has success, output, error — the ToolResult shape."""
    result = RelationalLens().query(kernel=InMemoryKernel())
    assert hasattr(result, "success")
    assert hasattr(result, "output")
    assert hasattr(result, "error")


def test_query_filter_by_payload_predicate() -> None:
    """filter= predicate selects only matching rows."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"match"), CellWriteEvent(payload=b"skip")])
    lens = RelationalLens()

    result = lens.query(kernel=kernel, filter=lambda row: row["payload"] == b"match")

    assert result.success is True
    assert len(result.output) == 1
    assert result.output[0]["payload"] == b"match"


def test_query_filter_by_cell_ref() -> None:
    """filter= can select a row by its cell_ref (hash)."""
    kernel = InMemoryKernel()
    target = CellWriteEvent(payload=b"target")
    kernel.append_batch([target, CellWriteEvent(payload=b"other")])
    lens = RelationalLens()

    target_ref = target.cell_ref()
    result = lens.query(kernel=kernel, filter=lambda row: row["cell_ref"] == target_ref)

    assert len(result.output) == 1
    assert result.output[0]["cell_ref"] == target_ref


def test_query_project_selects_columns() -> None:
    """project= returns only the named columns in each output row."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"data")])
    lens = RelationalLens()

    result = lens.query(kernel=kernel, project=["cell_ref", "payload"])

    assert result.success is True
    row = result.output[0]
    assert set(row.keys()) == {"cell_ref", "payload"}
    assert row["payload"] == b"data"


def test_query_project_verbatim_payload_in_selected_columns() -> None:
    """project= for 'payload' returns the verbatim bytes, not a summary."""
    raw = b"\x00\xff raw bytes"
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=raw)])
    lens = RelationalLens()

    result = lens.query(kernel=kernel, project=["payload"])
    assert result.output[0]["payload"] == raw


def test_query_project_empty_list_returns_empty_dicts() -> None:
    """project=[] means no columns selected — each output row is an empty dict."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])
    lens = RelationalLens()

    result = lens.query(kernel=kernel, project=[])

    assert result.success is True
    assert len(result.output) == 1
    assert result.output[0] == {}


def test_query_unknown_typed_key_raises_explicitly() -> None:
    """query() on an unknown projection key is an explicit error — not a silent empty."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])
    lens = RelationalLens()

    with pytest.raises(KeyError):
        lens.query(kernel=kernel, project=["cell_ref", "nonexistent_column"])


def test_query_unknown_key_raises_regardless_of_matching_rows() -> None:
    """Even an unknown key with a filter that selects zero rows must raise."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])
    lens = RelationalLens()

    with pytest.raises(KeyError):
        lens.query(
            kernel=kernel,
            filter=lambda row: False,  # selects no rows
            project=["bogus_field"],
        )


def test_query_filter_then_project_combined() -> None:
    """filter and project work together: filter first, then project columns."""
    kernel = InMemoryKernel()
    kernel.append_batch(
        [
            CellWriteEvent(payload=b"keep"),
            CellWriteEvent(payload=b"drop"),
        ]
    )
    lens = RelationalLens()

    result = lens.query(
        kernel=kernel,
        filter=lambda row: row["payload"] == b"keep",
        project=["payload", "seq_pos"],
    )

    assert len(result.output) == 1
    row = result.output[0]
    assert set(row.keys()) == {"payload", "seq_pos"}
    assert row["payload"] == b"keep"


def test_query_no_rows_filter_matches_nothing() -> None:
    """A filter that matches nothing returns success with an empty output."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"x")])
    lens = RelationalLens()

    result = lens.query(kernel=kernel, filter=lambda row: False)

    assert result.success is True
    assert result.output == []
