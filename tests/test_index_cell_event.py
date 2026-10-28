"""Tests for IndexCellEvent (D2a: index cell defining event).

Design Decision D2a: IndexCellEvent is the dedicated defining event for an
IndexCell produced by the static fold of access events.  Its replay.py branch
mirrors InterpreterOutputEvent exactly: target_ref() returns cell_ref,
is_defining() checks cell_ref, and the fold sets payload = ev.output.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from amplifier_data.models import (
    Event,
    IndexCellEvent,
)
from amplifier_data.replay import (
    is_defining,
    regenerate,
    target_ref,
)

# ---------------------------------------------------------------------------
# Model: IndexCellEvent structure
# ---------------------------------------------------------------------------


def test_index_cell_event_is_an_event_and_frozen() -> None:
    """IndexCellEvent extends Event and is immutable."""
    ev = IndexCellEvent(
        cell_ref="abc123",
        access_summary=b"summary bytes",
        output=b"index bytes",
    )
    assert isinstance(ev, Event)
    with pytest.raises(FrozenInstanceError):
        ev.output = b"mutated"  # type: ignore[misc]


def test_index_cell_event_fields_hold_values() -> None:
    """IndexCellEvent exposes cell_ref, access_summary, and output."""
    ev = IndexCellEvent(
        cell_ref="deadbeef" * 8,
        access_summary=b"the folded access summary",
        output=b"the index bytes",
    )
    assert ev.cell_ref == "deadbeef" * 8
    assert ev.access_summary == b"the folded access summary"
    assert ev.output == b"the index bytes"


def test_index_cell_derive_ref_is_stable() -> None:
    """Same inputs → same hash; different inputs → different hash."""
    ref_a = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"index")
    ref_b = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"index")
    ref_c = IndexCellEvent.derive_ref(access_summary=b"different", output=b"index")
    assert ref_a == ref_b
    assert ref_a != ref_c
    assert isinstance(ref_a, str)  # hexdigest


def test_index_cell_derive_ref_excludes_cell_ref_field() -> None:
    """The stored cell_ref equals the derived identity (no circularity)."""
    ref = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"index bytes")
    ev = IndexCellEvent(
        cell_ref=ref,
        access_summary=b"summary",
        output=b"index bytes",
    )
    assert ev.cell_ref == ref
    # Re-deriving yields the same hash:
    same = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"index bytes")
    assert same == ref


# ---------------------------------------------------------------------------
# Replay: target_ref / is_defining
# ---------------------------------------------------------------------------


def test_target_ref_returns_cell_ref_for_index_cell_event() -> None:
    """target_ref() returns the cell_ref from IndexCellEvent."""
    ref = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"idx")
    ev = IndexCellEvent(cell_ref=ref, access_summary=b"summary", output=b"idx")
    assert target_ref(ev) == ref


def test_is_defining_true_for_matching_cell_ref() -> None:
    """is_defining() returns True when cell_ref matches."""
    ref = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"idx")
    ev = IndexCellEvent(cell_ref=ref, access_summary=b"summary", output=b"idx")
    assert is_defining(ev, ref) is True


def test_is_defining_false_for_mismatched_cell_ref() -> None:
    """is_defining() returns False when cell_ref does not match."""
    ref = IndexCellEvent.derive_ref(access_summary=b"summary", output=b"idx")
    ev = IndexCellEvent(cell_ref=ref, access_summary=b"summary", output=b"idx")
    assert is_defining(ev, "totally-different-ref") is False


# ---------------------------------------------------------------------------
# Replay: regenerate — fold produces byte-identical output
# ---------------------------------------------------------------------------


def test_regenerate_folds_index_cell_event_payload() -> None:
    """regenerate() sets payload = ev.output for IndexCellEvent (D2a fold)."""
    access_summary = b"access summary bytes"
    index_bytes = b"the serialised index"
    ref = IndexCellEvent.derive_ref(access_summary=access_summary, output=index_bytes)
    ev = IndexCellEvent(
        cell_ref=ref,
        access_summary=access_summary,
        output=index_bytes,
    )
    cell = regenerate([(0, ev)], ref)
    assert cell.payload == index_bytes
    assert cell.provenance == (ev,)


def test_regenerate_index_cell_byte_identical_with_provenance() -> None:
    """Regenerated IndexCell is byte-identical; provenance carries the event."""
    access_summary = b"who read what"
    index_bytes = b"serialised index blob"
    ref = IndexCellEvent.derive_ref(access_summary=access_summary, output=index_bytes)
    ev = IndexCellEvent(
        cell_ref=ref,
        access_summary=access_summary,
        output=index_bytes,
    )

    first = regenerate([(0, ev)], ref)
    second = regenerate([(0, ev)], ref)

    # Byte-identical: same input → same output
    assert first.payload == second.payload == index_bytes
    # Provenance present: the defining event is in the provenance
    assert ev in first.provenance
    assert isinstance(first.provenance[0], IndexCellEvent)
    # access_summary is accessible via provenance
    assert first.provenance[0].access_summary == access_summary
