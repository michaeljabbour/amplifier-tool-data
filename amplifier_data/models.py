"""Frozen data models for the amplifier-data floor: events and the Cell."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# A sha256 hexdigest string identifying a cell or a content-addressed payload.
Hash = str
# A monotonically increasing position (u64) assigned by append_batch.
SeqPos = int


def _canonical(d: dict[str, object]) -> bytes:
    """Deterministic JSON bytes for hashing a defining event."""
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_address(payload: bytes) -> Hash:
    """sha256 hexdigest of arbitrary bytes."""
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Event:
    """Base marker for every event in the append-only log. Immutable."""


@dataclass(frozen=True)
class CellWriteEvent(Event):
    """Defines a cell: verbatim payload plus the interpreter refs that can read it."""

    payload: bytes
    interpreters: tuple[Hash, ...] = ()

    def cell_ref(self) -> Hash:
        """Stable identity of the cell this event defines (its content address)."""
        return content_address(
            _canonical(
                {
                    "kind": "cell_write",
                    "payload": self.payload.hex(),
                    "interpreters": list(self.interpreters),
                }
            )
        )


@dataclass(frozen=True)
class RelationshipEvent(Event):
    """An edge. Lands in the provenance of `from_ref` ONLY (not `to_ref`)."""

    from_ref: Hash
    to_ref: Hash
    type: str


@dataclass(frozen=True)
class ToolResultEvent(Event):
    """A verbatim external tool result, referenced by hash from an interpreter output."""

    output: bytes

    @property
    def hash(self) -> Hash:
        return content_address(self.output)


@dataclass(frozen=True)
class InterpreterOutputEvent(Event):
    """Defines a derived cell from a recorded interpreter run (F1)."""

    cell_ref: Hash  # identity of the derived cell this event defines
    model: str
    input_hash: Hash
    tool_results_consumed: tuple[Hash, ...]
    output: bytes

    @staticmethod
    def derive_ref(
        model: str,
        input_hash: Hash,
        tool_results_consumed: tuple[Hash, ...],
        output: bytes,
    ) -> Hash:
        """Identity of the derived cell — over the defining content, excluding cell_ref."""
        return content_address(
            _canonical(
                {
                    "kind": "interpreter_output",
                    "model": model,
                    "input_hash": input_hash,
                    "tool_results_consumed": list(tool_results_consumed),
                    "output": output.hex(),
                }
            )
        )


@dataclass(frozen=True)
class AccessEvent(Event):
    """A free-floating log entry recording that a cell was read (D3: provenance-neutral).

    AccessEvent is NOT associated with any cell's provenance — target_ref returns None
    for it, so it is never selected by regenerate().  It exists in the totally-ordered
    log purely for the IndexLens to fold.
    """

    cell_ref: Hash  # which cell was read


@dataclass(frozen=True)
class IndexCellEvent(Event):
    """Defines an index cell from a static fold over access events (D2a).

    The index cell is the output of the IndexLens 'static fold' — a derived cell
    whose payload is the serialised index and whose provenance carries the
    access_summary ('why I indexed this') for reproducibility.

    Replay: target_ref() returns cell_ref; is_defining() checks cell_ref;
    the fold sets payload = ev.output — mirroring InterpreterOutputEvent exactly.
    """

    cell_ref: Hash  # identity of the index cell this event defines
    access_summary: bytes  # the folded access summary ('why I indexed this' provenance)
    output: bytes  # the serialised index bytes produced by the static fold

    @staticmethod
    def derive_ref(access_summary: bytes, output: bytes) -> Hash:
        """Identity of the index cell — over the defining content, excluding cell_ref."""
        return content_address(
            _canonical(
                {
                    "kind": "index_cell",
                    "access_summary": access_summary.hex(),
                    "output": output.hex(),
                }
            )
        )


@dataclass(frozen=True)
class Cell:
    """A regenerable materialized view: verbatim payload + ordered provenance selection."""

    payload: bytes
    provenance: tuple[Event, ...]
    interpreters: tuple[Hash, ...] = ()
