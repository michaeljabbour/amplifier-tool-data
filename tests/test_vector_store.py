"""Tests for AmplifierStore.add_embedding / query_vector \u2014 Plan3c (Tasks 7 & 8).

Design Decision D4 (read-vs-fold boundary)
--------------------------------------------
add_embedding is a WRITE method (packs + write_cell + relate) and must
NOT emit any AccessEvent.  query_vector delegates to VectorLens which reads
kernel.all_events() directly \u2014 also no AccessEvent (D4 read-vs-fold boundary).
"""

from __future__ import annotations

import struct

import pytest

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses.vector import EMBEDDING_OF
from amplifier_data.models import AccessEvent, CellWriteEvent, RelationshipEvent
from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store() -> AmplifierStore:
    """Return an in-memory store (rust-free, no compiled extension needed)."""
    return AmplifierStore(kernel=InMemoryKernel())


# ---------------------------------------------------------------------------
# Tests \u2014 add_embedding (Task 7)
# ---------------------------------------------------------------------------


def test_add_embedding_returns_content_address_of_packed_vector() -> None:
    """emb_ref must equal CellWriteEvent(payload=pack('<Nf', *vector)).cell_ref()."""
    store = _store()
    target_ref = store.write_cell(b"some target content")
    vector = [1.0, 2.0, 3.0]

    emb_ref = store.add_embedding(target_ref, vector)

    expected_payload = struct.pack(f"<{len(vector)}f", *vector)
    expected_ref = CellWriteEvent(payload=expected_payload).cell_ref()
    assert emb_ref == expected_ref


def test_add_embedding_appends_cellwrite_and_embedding_of_edge() -> None:
    """Kernel log must contain CellWriteEvent(packed vector) + RelationshipEvent(EMBEDDING_OF)."""
    store = _store()
    target_ref = store.write_cell(b"target")
    vector = [0.5, 0.5]

    emb_ref = store.add_embedding(target_ref, vector)

    all_events = [ev for _, ev in store.kernel.all_events()]  # type: ignore[attr-defined]

    # A CellWriteEvent with the packed vector payload must be present.
    packed = struct.pack("<2f", 0.5, 0.5)
    cell_write_events = [e for e in all_events if isinstance(e, CellWriteEvent)]
    assert any(e.payload == packed for e in cell_write_events), (
        "No CellWriteEvent with packed vector payload found"
    )

    # A RelationshipEvent of type EMBEDDING_OF from emb_ref to target_ref must be present.
    rel_events = [e for e in all_events if isinstance(e, RelationshipEvent)]
    assert any(
        e.type == EMBEDDING_OF and e.from_ref == emb_ref and e.to_ref == target_ref
        for e in rel_events
    ), f"No RelationshipEvent(type={EMBEDDING_OF!r}, from={emb_ref!r}, to={target_ref!r}) found"


def test_add_embedding_does_not_emit_access_events() -> None:
    """add_embedding is a write operation \u2014 it must NOT append any AccessEvent (D4)."""
    store = _store()
    target_ref = store.write_cell(b"data")
    vector = [0.1, 0.2, 0.3, 0.4]

    store.add_embedding(target_ref, vector)

    all_events = store.kernel.all_events()  # type: ignore[attr-defined]
    access_events = [ev for _, ev in all_events if isinstance(ev, AccessEvent)]
    assert access_events == [], f"add_embedding must not emit AccessEvents; found: {access_events}"


# ---------------------------------------------------------------------------
# Tests \u2014 query_vector (Task 8)
# ---------------------------------------------------------------------------


def test_query_vector_returns_top_k_targets_in_order() -> None:
    """Top-k results must be sorted by descending cosine similarity."""
    store = _store()
    near = store.write_cell(b"near")
    orth = store.write_cell(b"orth")
    store.add_embedding(near, (1.0, 0.0))
    store.add_embedding(orth, (0.0, 1.0))

    out = store.query_vector((1.0, 0.0), k=2)

    assert [ref for ref, _ in out] == [near, orth]
    assert out[0][1] == pytest.approx(1.0)
    assert out[1][1] == pytest.approx(0.0)


def test_query_vector_empty_store_returns_empty_list() -> None:
    """query_vector on a store with no embeddings must return an empty list."""
    assert _store().query_vector((1.0, 0.0), k=5) == []


def test_query_vector_appends_no_access_events() -> None:
    """query_vector reads the kernel directly \u2014 no AccessEvents appended (D4 boundary)."""
    from amplifier_data.models import AccessEvent

    store = _store()
    t = store.write_cell(b"t")
    store.add_embedding(t, (1.0, 0.0))
    before = len(store.kernel.all_events())  # type: ignore[attr-defined]
    store.query_vector((1.0, 0.0), k=1)
    after = store.kernel.all_events()  # type: ignore[attr-defined]
    assert len(after) == before, "query_vector must not append events (D4 read-vs-fold boundary)"
    assert not any(isinstance(ev, AccessEvent) for _pos, ev in after)
