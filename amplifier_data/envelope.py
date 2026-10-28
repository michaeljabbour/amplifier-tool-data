"""Atomic multi-fact batches and a compensating-write rollback envelope (T3D-2).

Two seams, two failure models
------------------------------
1. :class:`WriteBatch` — *single-call* atomicity. Stage several events (e.g. a
   memory cell plus its ``importance`` + ``category`` + ``source`` facts) and
   land them in ONE ``kernel.append_batch`` call. The kernel validates the whole
   batch before mutating anything, so a malformed batch leaves NO half-write:
   either every staged event lands at a contiguous run of SeqPos, or none does.

2. :class:`Transaction` — *multi-call* rollback. Some flows must span several
   append_batch calls (a write whose follow-up facts depend on a ref read back
   from an earlier commit). The log is append-only, so "rollback" is by
   COMPENSATION: on exception, the envelope replays a reversing write for each
   committed fact/edge (``invalidate_fact`` / an ``!``-prefixed tombstone edge).
   Cell *payloads* are immutable and content-addressed — they cannot be deleted,
   but an orphaned cell carries no asserted facts and is reclaimed by
   ``compact_into`` (which drops invalidated facts + tombstones). So the
   externally-visible state after a rolled-back transaction equals the state
   before it began.

Why compensation rather than true delete: see Conventions in AGENTS.md —
"The log is truth. Every Cell must be regenerable from the event log alone."
A destructive rollback would violate that; a compensating write preserves it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .causal import causal_edges
from .lenses._scope import SCOPED_TO
from .lenses.temporal import INVALIDATE_PREFIX
from .models import CellWriteEvent, Event, Hash, RelationshipEvent, SeqPos

if TYPE_CHECKING:
    from collections.abc import Callable

    from .store import AmplifierStore


class WriteBatch:
    """Stage writes, then :meth:`commit` them as ONE all-or-nothing append_batch.

    Refs are returned eagerly (content addressing makes them knowable before the
    commit), so callers can wire relationships between staged cells in the same
    batch. Nothing touches the log until :meth:`commit`.
    """

    def __init__(self, store: AmplifierStore) -> None:
        self._store = store
        self._events: list[Event] = []

    def write_cell(
        self,
        payload: bytes,
        interpreters: tuple[Hash, ...] = (),
        *,
        source_interaction_id: str | None = None,
        written_by_hook: str | None = None,
    ) -> Hash:
        """Stage a cell write (plus optional causal edges); return its ref now."""
        event = CellWriteEvent(payload=payload, interpreters=interpreters)
        ref = event.cell_ref()
        self._events.append(event)
        self._events.extend(
            causal_edges(
                ref,
                source_interaction_id=source_interaction_id,
                written_by_hook=written_by_hook,
            )
        )
        return ref

    def relate(self, from_ref: Hash, to_ref: Hash, rel_type: str) -> WriteBatch:
        """Stage an edge; returns self for chaining."""
        self._events.append(RelationshipEvent(from_ref=from_ref, to_ref=to_ref, type=rel_type))
        return self

    def assert_fact(self, subject: Hash, predicate: str, object: Hash) -> WriteBatch:
        """Stage a KG fact (sugar over :meth:`relate`)."""
        return self.relate(subject, object, predicate)

    def scope(self, cell_ref: Hash, scope_ref: Hash) -> WriteBatch:
        """Stage a scope edge (``SCOPED_TO``)."""
        return self.relate(cell_ref, scope_ref, SCOPED_TO)

    def add_event(self, event: Event) -> WriteBatch:
        """Stage a raw pre-built event (escape hatch for advanced callers)."""
        self._events.append(event)
        return self

    @property
    def staged(self) -> list[Event]:
        """A copy of the staged events (inspection / testing)."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def commit(self) -> list[SeqPos]:
        """Append every staged event in a single atomic batch; return SeqPos run.

        A no-op (returns ``[]``) when nothing was staged. On any validation
        failure the underlying kernel raises and the log is unchanged.
        """
        if not self._events:
            return []
        return self._store.kernel.append_batch(list(self._events))


class Transaction:
    """Multi-call write envelope with compensating-write rollback.

    Use as a context manager::

        with store.transaction() as txn:
            ref = txn.write_cell(payload, source_interaction_id="abc")
            txn.assert_fact(ref, "importance", high_ref)
            txn.assert_fact(ref, "category", note_ref)
            ...  # if anything here raises, both facts are compensated away

    On a clean exit the writes simply stand. On an exception every fact/edge
    written through the envelope is reversed in LIFO order, then the original
    exception propagates.
    """

    def __init__(self, store: AmplifierStore) -> None:
        self._store = store
        self._compensators: list[Callable[[], None]] = []

    def __enter__(self) -> Transaction:
        return self

    def write_cell(
        self,
        payload: bytes,
        interpreters: tuple[Hash, ...] = (),
        *,
        source_interaction_id: str | None = None,
        written_by_hook: str | None = None,
    ) -> Hash:
        """Write a cell immediately. Payload is immutable, so there is nothing to
        compensate: a rolled-back cell is left orphaned (no facts point at it) and
        is reclaimed by compaction."""
        return self._store.write_cell(
            payload,
            interpreters,
            source_interaction_id=source_interaction_id,
            written_by_hook=written_by_hook,
        )

    def assert_fact(self, subject: Hash, predicate: str, object: Hash) -> None:
        """Assert a fact and register its compensator (:meth:`invalidate_fact`)."""
        self._store.assert_fact(subject, predicate, object)
        self._compensators.append(lambda: self._store.invalidate_fact(subject, predicate, object))

    def relate(self, from_ref: Hash, to_ref: Hash, rel_type: str) -> None:
        """Write an edge and register a tombstone compensator."""
        self._store.relate(from_ref, to_ref, rel_type)
        self._compensators.append(
            lambda: self._store.relate(from_ref, to_ref, INVALIDATE_PREFIX + rel_type)
        )

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is not None:
            # Reverse in LIFO order so later writes are undone before earlier ones.
            for compensate in reversed(self._compensators):
                compensate()
        self._compensators.clear()
        return False  # never suppress the original exception
