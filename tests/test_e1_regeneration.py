import pytest

from amplifier_data.models import (
    CellWriteEvent,
    Event,
    InterpreterOutputEvent,
    SeqPos,
)
from amplifier_data.replay import MissingDefiningEvent, regenerate


def test_regenerate_folds_a_written_cell() -> None:
    write = CellWriteEvent(payload=b"verbatim", interpreters=("agent-1",))
    events: list[tuple[SeqPos, Event]] = [(0, write)]
    cell = regenerate(events, write.cell_ref())
    assert cell.payload == b"verbatim"
    assert cell.interpreters == ("agent-1",)
    assert cell.provenance == (write,)


def test_regenerate_folds_a_derived_interpreter_cell() -> None:
    ref = InterpreterOutputEvent.derive_ref(
        model="m", input_hash="ih", tool_results_consumed=(), output=b"recorded output"
    )
    interp = InterpreterOutputEvent(
        cell_ref=ref,
        model="m",
        input_hash="ih",
        tool_results_consumed=(),
        output=b"recorded output",
    )
    cell = regenerate([(0, interp)], ref)
    assert cell.payload == b"recorded output"  # derived payload == recorded output
    assert cell.provenance == (interp,)


def test_regenerate_unknown_ref_raises_missing_defining_event() -> None:
    write = CellWriteEvent(payload=b"x")
    with pytest.raises(MissingDefiningEvent):
        regenerate([(0, write)], "ref-that-has-no-defining-event")


def test_relationship_event_amends_from_ref_only() -> None:
    from amplifier_data.models import RelationshipEvent

    a = CellWriteEvent(payload=b"cell A")
    b = CellWriteEvent(payload=b"cell B")
    a_ref, b_ref = a.cell_ref(), b.cell_ref()
    edge = RelationshipEvent(from_ref=a_ref, to_ref=b_ref, type="links_to")
    events: list[tuple[int, Event]] = [(0, a), (1, b), (2, edge)]

    cell_a = regenerate(events, a_ref)
    assert cell_a.payload == b"cell A"  # edge does NOT change payload
    assert edge in cell_a.provenance  # edge lands in A's provenance
    assert cell_a.provenance == (a, edge)  # ordered by SeqPos

    cell_b = regenerate(events, b_ref)
    assert edge not in cell_b.provenance  # to_ref is NOT amended
    assert cell_b.provenance == (b,)


def test_store_e1_round_trip() -> None:
    from amplifier_data.store import AmplifierStore

    store = AmplifierStore()
    payload = b"verbatim drawer contents"
    ref = store.write_cell(payload)

    first = store.get_cell(ref)
    assert first.payload == payload

    store.drop_materialized(ref)  # discard the materialized cache
    regenerated = store.regenerate(ref)  # fold from the log ALONE
    assert regenerated.payload == payload  # identical bytes
    assert len(regenerated.provenance) == 1
    assert isinstance(regenerated.provenance[0], CellWriteEvent)


def test_store_relationship_amends_source_cell_only() -> None:
    from amplifier_data.models import RelationshipEvent
    from amplifier_data.store import AmplifierStore

    store = AmplifierStore()
    a = store.write_cell(b"cell A")
    b = store.write_cell(b"cell B")
    store.relate(a, b, "links_to")

    cell_a = store.regenerate(a)
    assert cell_a.payload == b"cell A"
    edges = [e for e in cell_a.provenance if isinstance(e, RelationshipEvent)]
    assert len(edges) == 1
    assert edges[0].to_ref == b

    cell_b = store.regenerate(b)
    assert not any(isinstance(e, RelationshipEvent) for e in cell_b.provenance)
