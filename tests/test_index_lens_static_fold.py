"""Tests for IndexLens static reflexive fold (D2/D3/D4 compliance).

IndexLens performs an EXPLICIT STATIC FOLD over the recorded AccessEvents
and produces the index Cell's DEFINING EVENT (IndexCellEvent, per D2).
The index Cell materialises and regenerate()s to a BYTE-IDENTICAL IndexCell.
No mutable side-car; reads via kernel.all_events() directly.
"""

from __future__ import annotations

import json

import pytest

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses import Lens
from amplifier_data.lenses.index import IndexLens
from amplifier_data.models import AccessEvent, Cell, CellWriteEvent, IndexCellEvent
from amplifier_data.replay import MissingDefiningEvent, regenerate

# ---------------------------------------------------------------------------
# Lens Protocol compliance
# ---------------------------------------------------------------------------


def test_index_lens_satisfies_lens_protocol() -> None:
    """IndexLens satisfies the Lens runtime_checkable Protocol."""
    lens = IndexLens()
    assert isinstance(lens, Lens)


def test_index_lens_name_is_index() -> None:
    """IndexLens.name is 'index'."""
    lens = IndexLens()
    assert lens.name == "index"


# ---------------------------------------------------------------------------
# project() — returns Cell (not dict), D2 compliance
# ---------------------------------------------------------------------------


def test_project_returns_cell() -> None:
    """project() returns a Cell, not a plain dict."""
    kernel = InMemoryKernel()
    lens = IndexLens()
    result = lens.project(kernel)
    assert isinstance(result, Cell), (
        "IndexLens.project() must return a Cell — the defining event materialised "
        f"via replay.regenerate(); got {type(result).__name__}"
    )


def test_project_cell_provenance_contains_index_cell_event() -> None:
    """The Cell from project() has an IndexCellEvent in its provenance (D2)."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"hello")
    kernel.append_batch([cwe])
    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    lens = IndexLens()
    cell = lens.project(kernel)

    assert len(cell.provenance) >= 1
    assert isinstance(cell.provenance[0], IndexCellEvent), (
        "First provenance entry must be an IndexCellEvent (the defining event)"
    )


def test_project_index_cell_event_carries_access_summary() -> None:
    """IndexCellEvent.access_summary encodes the folded access counts (provenance)."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])
    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    cell = IndexLens().project(kernel)
    ice = cell.provenance[0]

    assert isinstance(ice, IndexCellEvent)
    summary = json.loads(ice.access_summary.decode("utf-8"))
    assert cwe.cell_ref() in summary, "access_summary must include the read cell_ref"
    assert summary[cwe.cell_ref()] == 1


def test_project_cell_payload_is_access_count_json() -> None:
    """Cell payload is a deterministic JSON encoding of the access count table."""
    kernel = InMemoryKernel()
    cwe_a = CellWriteEvent(payload=b"A")
    cwe_b = CellWriteEvent(payload=b"B")
    kernel.append_batch([cwe_a, cwe_b])
    for _ in range(3):
        kernel.append_batch([AccessEvent(cell_ref=cwe_a.cell_ref())])
    kernel.append_batch([AccessEvent(cell_ref=cwe_b.cell_ref())])

    cell = IndexLens().project(kernel)
    counts: dict[str, int] = json.loads(cell.payload.decode("utf-8"))

    assert counts[cwe_a.cell_ref()] == 3
    assert counts[cwe_b.cell_ref()] == 1


def test_project_empty_kernel_returns_cell_with_empty_index() -> None:
    """project() on an empty kernel returns a Cell with an empty index payload."""
    kernel = InMemoryKernel()
    cell = IndexLens().project(kernel)

    assert isinstance(cell, Cell)
    counts: dict[str, int] = json.loads(cell.payload.decode("utf-8"))
    assert counts == {}


# ---------------------------------------------------------------------------
# Byte-identical regeneration guarantee
# ---------------------------------------------------------------------------


def test_project_byte_identical_on_same_kernel() -> None:
    """Calling project() twice on the same kernel produces byte-identical Cells."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])
    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    lens = IndexLens()
    cell1 = lens.project(kernel)
    cell2 = lens.project(kernel)

    assert cell1.payload == cell2.payload, "payload must be byte-identical"
    assert cell1.provenance == cell2.provenance, "provenance must be identical"


def test_project_cell_regenerates_from_defining_event() -> None:
    """The Cell can be byte-identically regenerated from its provenance IndexCellEvent."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])
    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    original = IndexLens().project(kernel)
    ice = original.provenance[0]
    assert isinstance(ice, IndexCellEvent)

    # Regenerate from the defining event in a synthetic single-event log
    regenerated = regenerate([(0, ice)], ice.cell_ref)

    assert regenerated.payload == original.payload, "regenerated payload must be byte-identical"
    assert regenerated.provenance == original.provenance, "provenance must be identical"


def test_project_cell_changes_when_new_access_events_added() -> None:
    """project() re-folds on each call: new AccessEvents produce a different Cell."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])

    lens = IndexLens()
    cell_before = lens.project(kernel)
    counts_before: dict[str, int] = json.loads(cell_before.payload.decode("utf-8"))

    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    cell_after = lens.project(kernel)
    counts_after: dict[str, int] = json.loads(cell_after.payload.decode("utf-8"))

    assert counts_before == {}
    assert counts_after[cwe.cell_ref()] == 1
    assert cell_before.payload != cell_after.payload, "byte content must change after new reads"


# ---------------------------------------------------------------------------
# No mutable side-car — project() does NOT mutate the kernel (D4)
# ---------------------------------------------------------------------------


def test_project_does_not_append_to_kernel() -> None:
    """project() does not append any events to the kernel (pure static fold)."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])
    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    count_before = len(kernel.all_events())
    IndexLens().project(kernel)
    count_after = len(kernel.all_events())

    assert count_after == count_before, (
        f"project() must not mutate the kernel; "
        f"event count changed from {count_before} to {count_after}"
    )


def test_two_lens_instances_are_independent() -> None:
    """Two IndexLens instances over the same kernel produce equal, independent views."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])
    kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    cell_a = IndexLens().project(kernel)
    cell_b = IndexLens().project(kernel)

    assert cell_a.payload == cell_b.payload


# ---------------------------------------------------------------------------
# query() — backward-compatible typed surface returning dict[Hash, int]
# ---------------------------------------------------------------------------


def test_query_returns_successful_query_result() -> None:
    """query() returns QueryResult with success=True."""
    kernel = InMemoryKernel()
    result = IndexLens().query(kernel=kernel)
    assert result.success is True


def test_query_output_is_dict() -> None:
    """query().output is a dict (count table)."""
    kernel = InMemoryKernel()
    result = IndexLens().query(kernel=kernel)
    assert isinstance(result.output, dict)


def test_query_empty_kernel_empty_output() -> None:
    """query() on an empty kernel returns an empty count dict."""
    kernel = InMemoryKernel()
    result = IndexLens().query(kernel=kernel)
    assert result.output == {}


def test_query_counts_access_events() -> None:
    """query() reflects AccessEvents: three reads of the same cell → count 3."""
    kernel = InMemoryKernel()
    cwe = CellWriteEvent(payload=b"data")
    kernel.append_batch([cwe])
    for _ in range(3):
        kernel.append_batch([AccessEvent(cell_ref=cwe.cell_ref())])

    result = IndexLens().query(kernel=kernel)

    assert result.success
    assert result.output.get(cwe.cell_ref()) == 3


def test_query_cell_ref_filter_restricts_to_one_cell() -> None:
    """query(cell_ref=ref) restricts output to that single cell."""
    kernel = InMemoryKernel()
    cwe_a = CellWriteEvent(payload=b"A")
    cwe_b = CellWriteEvent(payload=b"B")
    kernel.append_batch([cwe_a, cwe_b])
    kernel.append_batch([AccessEvent(cell_ref=cwe_a.cell_ref())])
    kernel.append_batch([AccessEvent(cell_ref=cwe_b.cell_ref())])
    kernel.append_batch([AccessEvent(cell_ref=cwe_b.cell_ref())])

    result = IndexLens().query(kernel=kernel, cell_ref=cwe_b.cell_ref())

    assert result.success
    assert result.output[cwe_b.cell_ref()] == 2
    assert cwe_a.cell_ref() not in result.output


def test_query_cell_ref_never_read_returns_zero() -> None:
    """query(cell_ref=<never-read>) returns {cell_ref: 0}."""
    kernel = InMemoryKernel()
    result = IndexLens().query(kernel=kernel, cell_ref="never-read-ref")
    assert result.success
    assert result.output == {"never-read-ref": 0}


def test_query_missing_kernel_raises_type_error() -> None:
    """query() without 'kernel' keyword argument raises TypeError."""
    lens = IndexLens()
    with pytest.raises(TypeError, match="kernel"):
        lens.query()


# ---------------------------------------------------------------------------
# Error handling: MissingDefiningEvent when defining event absent from log
# ---------------------------------------------------------------------------


def test_missing_index_cell_event_raises_missing_defining_event() -> None:
    """regenerate() raises MissingDefiningEvent (with specific cell_ref) when absent."""
    # Build an IndexCellEvent in memory (not in the kernel)
    ice = IndexCellEvent(
        cell_ref=IndexCellEvent.derive_ref(b"some summary", b"some output"),
        access_summary=b"some summary",
        output=b"some output",
    )

    # Regenerate from an EMPTY event list — no defining event present
    with pytest.raises(MissingDefiningEvent) as exc_info:
        regenerate([], ice.cell_ref)

    # Error carries the SPECIFIC missing cell_ref (not a generic message)
    assert exc_info.value.cell_ref == ice.cell_ref, (
        f"MissingDefiningEvent must carry the specific cell_ref; got {exc_info.value.cell_ref!r}"
    )
