import hashlib
from dataclasses import FrozenInstanceError

import pytest

from amplifier_data.models import AccessEvent, CellWriteEvent, Event, content_address


def test_content_address_is_sha256_hexdigest() -> None:
    assert content_address(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert content_address(b"abc") == content_address(b"abc")
    assert content_address(b"abc") != content_address(b"abd")


def test_cell_write_event_is_an_event_and_frozen() -> None:
    e = CellWriteEvent(payload=b"hello")
    assert isinstance(e, Event)
    with pytest.raises(FrozenInstanceError):
        e.payload = b"world"  # type: ignore[misc]


def test_cell_write_ref_is_stable_and_content_derived() -> None:
    a = CellWriteEvent(payload=b"hello")
    b = CellWriteEvent(payload=b"hello")
    c = CellWriteEvent(payload=b"world")
    assert a.cell_ref() == b.cell_ref()  # same content -> same identity
    assert a.cell_ref() != c.cell_ref()  # different content -> different identity
    assert isinstance(a.cell_ref(), str)  # hexdigest string


def test_relationship_event_fields() -> None:
    from amplifier_data.models import RelationshipEvent

    r = RelationshipEvent(from_ref="aaa", to_ref="bbb", type="links_to")
    assert r.from_ref == "aaa"
    assert r.to_ref == "bbb"
    assert r.type == "links_to"


def test_tool_result_hash_is_content_address() -> None:
    from amplifier_data.models import ToolResultEvent

    t = ToolResultEvent(output=b"tool payload")
    assert t.hash == content_address(b"tool payload")


def test_interpreter_derive_ref_excludes_cell_ref_field() -> None:
    from amplifier_data.models import InterpreterOutputEvent

    ref = InterpreterOutputEvent.derive_ref(
        model="m1", input_hash="ih", tool_results_consumed=("t1", "t2"), output=b"out"
    )
    # The stored cell_ref equals the derived identity (no circularity).
    ev = InterpreterOutputEvent(
        cell_ref=ref,
        model="m1",
        input_hash="ih",
        tool_results_consumed=("t1", "t2"),
        output=b"out",
    )
    assert ev.cell_ref == ref
    # Identity is over the defining content, NOT the cell_ref field:
    same = InterpreterOutputEvent.derive_ref(
        model="m1", input_hash="ih", tool_results_consumed=("t1", "t2"), output=b"out"
    )
    assert same == ref


def test_cell_is_frozen_with_three_fields() -> None:
    from amplifier_data.models import Cell

    write = CellWriteEvent(payload=b"data")
    cell = Cell(payload=b"data", provenance=(write,), interpreters=("agent-1",))
    assert cell.payload == b"data"
    assert cell.provenance == (write,)
    assert cell.interpreters == ("agent-1",)
    with pytest.raises(FrozenInstanceError):
        cell.payload = b"mutated"  # type: ignore[misc]


# --- AccessEvent (Design Decision D3: provenance-neutral) ---


def test_access_event_is_an_event_and_frozen() -> None:
    """AccessEvent is a frozen dataclass extending Event."""
    ev = AccessEvent(cell_ref="abc123")
    assert isinstance(ev, Event)
    with pytest.raises(FrozenInstanceError):
        ev.cell_ref = "mutated"  # type: ignore[misc]


def test_access_event_cell_ref_field_holds_hash() -> None:
    """AccessEvent.cell_ref stores a Hash (str)."""
    ref = "deadbeef" * 8  # 64-char hex string
    ev = AccessEvent(cell_ref=ref)
    assert ev.cell_ref == ref
    assert isinstance(ev.cell_ref, str)


def test_access_event_target_ref_is_none() -> None:
    """D3: AccessEvent is provenance-neutral — target_ref returns None."""
    from amplifier_data.replay import target_ref

    ev = AccessEvent(cell_ref="abc123")
    assert target_ref(ev) is None


def test_access_event_not_selected_by_regenerate() -> None:
    """Appending an AccessEvent does not affect a cell's regenerated provenance."""
    from amplifier_data.kernel import InMemoryKernel
    from amplifier_data.models import Cell

    kernel = InMemoryKernel()
    write = CellWriteEvent(payload=b"data")
    kernel.append_batch([write])
    ref = write.cell_ref()

    # Append an AccessEvent referencing the same cell.
    access = AccessEvent(cell_ref=ref)
    kernel.append_batch([access])

    # The regenerated cell's provenance contains ONLY the defining write event.
    cell = kernel.resolve(ref)
    assert isinstance(cell, Cell)
    assert cell.provenance == (write,)
