"""TemporalLens — KG facts (subject, predicate, object) with validity windows.

Mirrors ``lenses/_scope.py``: a pure fold over ``kernel.all_events()`` with no
persisted state and no AccessEvents appended (D1/D4). A *fact* is a triple
(subject, predicate, object) carried by a ``RelationshipEvent``:

  * ASSERTION   — ``RelationshipEvent(from_ref=subject, to_ref=object, type=predicate)``
  * INVALIDATION— ``RelationshipEvent(from_ref=subject, to_ref=object,
                     type=INVALIDATE_PREFIX + predicate)``  (a reserved tombstone
                     prefix, mirroring how SCOPED_TO / EMBEDDING_OF are reserved
                     type strings)

DETERMINISM IS LOAD-BEARING. Events carry NO wall-clock timestamp (auto-generated
time would break E1 byte-identical regeneration). Validity is ordered by LOGICAL
time — the kernel's SeqPos. A triple is CURRENTLY VALID iff its latest event (by
SeqPos) is an assertion; re-asserting after an invalidation makes it valid again.
Consumers that need wall-clock validity put the timestamp IN the payload as data
(deterministic, consumer-supplied) — the substrate never invents time.

The reserved internal edge types SCOPED_TO and EMBEDDING_OF are NOT user facts and
are excluded from the fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INVALIDATE_PREFIX: str = "__invalidate__:"
"""Reserved type prefix marking a fact INVALIDATION (tombstone). A predicate must
not begin with this prefix. Mirrors SCOPED_TO / EMBEDDING_OF as a reserved type."""


@dataclass(frozen=True)
class Fact:
    """A knowledge-graph fact: the triple (subject, predicate, object)."""

    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class TimelineEntry:
    """One assert/invalidate event in a subject's history, ordered by SeqPos."""

    seq_pos: Any
    op: str  # "assert" | "invalidate"
    predicate: str
    object: str


@dataclass(frozen=True)
class QueryResult:
    """ToolResult-shaped result from :meth:`TemporalLens.query` (reuse, not reinvent)."""

    success: bool
    output: list[Fact]
    error: str | None = None


Result = QueryResult


class TemporalLens:
    """Validity-window fold over the RelationshipEvent log implementing the Lens Protocol."""

    @property
    def name(self) -> str:
        """Human-readable identifier for this lens type."""
        return "temporal"

    def _reserved_types(self) -> frozenset[str]:
        """Reserved internal edge types that are NOT user facts."""
        from amplifier_data.lenses._scope import SCOPED_TO
        from amplifier_data.lenses.vector import EMBEDDING_OF

        return frozenset({SCOPED_TO, EMBEDDING_OF})

    def _fold(self, kernel: Any) -> dict[tuple[str, str, str], list[tuple[Any, str]]]:
        """Fold the log into ``{(subject, predicate, object): [(seq_pos, op), ...]}``.

        Pure (D1/D4). Assertions and tombstone-prefixed invalidations are paired
        per triple in SeqPos order. Reserved internal edges are skipped.
        """
        from amplifier_data.models import RelationshipEvent

        reserved = self._reserved_types()
        history: dict[tuple[str, str, str], list[tuple[Any, str]]] = {}
        for seq_pos, event in kernel.all_events():
            if not isinstance(event, RelationshipEvent):
                continue
            etype = event.type
            if etype.startswith(INVALIDATE_PREFIX):
                predicate = etype[len(INVALIDATE_PREFIX) :]
                op = "invalidate"
            else:
                if etype in reserved:
                    continue
                predicate = etype
                op = "assert"
            history.setdefault((event.from_ref, predicate, event.to_ref), []).append((seq_pos, op))
        return history

    def current_facts(
        self, kernel: Any, subject: str | None = None, predicate: str | None = None
    ) -> list[Fact]:
        """Return the currently-valid facts (latest event per triple is an assertion).

        Optionally filtered by *subject* and/or *predicate*. Output is sorted for
        determinism.
        """
        history = self._fold(kernel)
        out: list[Fact] = []
        for (subj, pred, obj), events in history.items():
            if max(events, key=lambda e: e[0])[1] != "assert":
                continue
            if subject is not None and subj != subject:
                continue
            if predicate is not None and pred != predicate:
                continue
            out.append(Fact(subject=subj, predicate=pred, object=obj))
        return sorted(out, key=lambda f: (f.subject, f.predicate, f.object))

    def timeline(self, kernel: Any, subject: str) -> list[TimelineEntry]:
        """Return *subject*'s assert/invalidate history, ordered by SeqPos."""
        history = self._fold(kernel)
        entries: list[TimelineEntry] = []
        for (subj, pred, obj), events in history.items():
            if subj != subject:
                continue
            for seq_pos, op in events:
                entries.append(TimelineEntry(seq_pos=seq_pos, op=op, predicate=pred, object=obj))
        return sorted(entries, key=lambda e: e.seq_pos)

    def query(self, **typed_args: Any) -> QueryResult:
        """Typed surface over :meth:`current_facts`.

        Keyword arguments: ``kernel`` (required), ``subject`` (optional),
        ``predicate`` (optional). Returns a ToolResult-shaped result whose
        ``output`` is the sorted list of currently-valid facts.
        """
        if "kernel" not in typed_args:
            raise TypeError("TemporalLens.query() requires a 'kernel' keyword argument")
        kernel = typed_args["kernel"]
        subject: str | None = typed_args.get("subject")
        predicate: str | None = typed_args.get("predicate")
        return QueryResult(success=True, output=self.current_facts(kernel, subject, predicate))
