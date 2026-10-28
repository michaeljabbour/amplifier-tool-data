"""The replay/fold engine — the E1 regeneration guarantee. Pure functions, no inference."""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    Cell,
    CellWriteEvent,
    Event,
    Hash,
    IndexCellEvent,
    InterpreterOutputEvent,
    RelationshipEvent,
    SeqPos,
)


class MissingDefiningEvent(Exception):
    """Raised when no defining event exists for a cell_ref — a broken floor, loud."""

    def __init__(self, cell_ref: Hash) -> None:
        super().__init__(f"no defining event found for cell_ref {cell_ref!r}")
        self.cell_ref = cell_ref


def target_ref(event: Event) -> Hash | None:
    """The cell_ref whose provenance this event belongs to, or None."""
    if isinstance(event, CellWriteEvent):
        return event.cell_ref()
    if isinstance(event, InterpreterOutputEvent):
        return event.cell_ref
    if isinstance(event, IndexCellEvent):
        return event.cell_ref
    if isinstance(event, RelationshipEvent):
        return event.from_ref  # edge lands in the SOURCE cell's provenance only
    return None


def is_defining(event: Event, cell_ref: Hash) -> bool:
    """True if `event` DEFINES the cell identified by `cell_ref`."""
    if isinstance(event, CellWriteEvent):
        return event.cell_ref() == cell_ref
    if isinstance(event, InterpreterOutputEvent):
        return event.cell_ref == cell_ref
    if isinstance(event, IndexCellEvent):
        return event.cell_ref == cell_ref
    return False


def regenerate(events: Sequence[tuple[SeqPos, Event]], cell_ref: Hash) -> Cell:
    """Deterministic reduce over the ordered selection of events targeting cell_ref."""
    selected = sorted(
        (pair for pair in events if target_ref(pair[1]) == cell_ref),
        key=lambda pair: pair[0],
    )
    if not any(is_defining(ev, cell_ref) for _pos, ev in selected):
        raise MissingDefiningEvent(cell_ref)

    payload: bytes | None = None
    interpreters: tuple[Hash, ...] = ()
    provenance: list[Event] = []
    for _pos, ev in selected:
        provenance.append(ev)
        if isinstance(ev, CellWriteEvent):
            payload = ev.payload
            interpreters = ev.interpreters
        elif isinstance(ev, InterpreterOutputEvent):
            payload = ev.output
        elif isinstance(ev, IndexCellEvent):
            payload = ev.output
    assert payload is not None  # guaranteed by the defining-event check above
    return Cell(payload=payload, provenance=tuple(provenance), interpreters=interpreters)
