"""Scope fold — pure view of SCOPED_TO membership edges in the event log.

Design Decision F3 — direct edges only
----------------------------------------
``fold_scope`` records ONLY the direct ``RelationshipEvent`` edges whose
``type == SCOPED_TO``.  It does **not** perform transitive closure or follow
scope-to-scope edges.  Rationale: the scope graph may be cyclic, sparse, or
partially loaded; computing transitive membership would couple the fold to
ordering assumptions and magnify subtle log inconsistencies into unbounded
expansion.  Callers that need transitive membership can walk the returned
``ScopeIndex.membership`` themselves with full visibility into the traversal.

Single-source rationale
-----------------------
All membership information is derived from ``kernel.all_events()`` on every
call (Design Decision D1).  The fold persists no state, appends no events,
and is safe to call concurrently from multiple callsites.

SCOPED_TO mirrors EMBEDDING_OF in ``lenses/vector.py``: one canonical
``str`` constant shared by the writer (``AmplifierStore.scope``) and the
reader (``fold_scope``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Canonical relationship-type constant
# ---------------------------------------------------------------------------

SCOPED_TO: str = "scoped_to"
"""Edge type: *from* a content cell *to* the scope cell it belongs to.

Used by ``AmplifierStore.scope`` when writing and consumed by
:func:`fold_scope` when folding.  Mirrors ``EMBEDDING_OF`` in
``lenses/vector.py``.
"""

# ---------------------------------------------------------------------------
# Fold result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeIndex:
    """Immutable result of :func:`fold_scope`: direct-edge scope membership.

    ``membership`` maps each content ``cell_ref`` to the ``frozenset`` of
    scope ``cell_ref`` values it is **directly** attached to (F3: no
    transitive closure).
    """

    membership: dict[str, frozenset[str]]

    def is_in_scope(self, cell_ref: str, scope_ref: str) -> bool:
        """Return True iff *cell_ref* has a direct SCOPED_TO edge to *scope_ref*.

        Design Decision F6 — unknown refs return False, never raise.

        Parameters
        ----------
        cell_ref:
            The content cell whose scope membership is being tested.
        scope_ref:
            The scope to test membership against.
        """
        return scope_ref in self.membership.get(cell_ref, frozenset())

    def cells_in_scope(self, scope_ref: str) -> frozenset[str]:
        """Return all cells directly scoped to *scope_ref*.

        Design Decision F6 — an unknown scope (no member cells) yields an
        empty ``frozenset``, never raises.

        Parameters
        ----------
        scope_ref:
            The scope whose member cells are requested.

        Returns
        -------
        frozenset[str]
            The set of ``cell_ref`` values whose direct SCOPED_TO edge
            points to *scope_ref*.  Empty if *scope_ref* is unknown.
        """
        return frozenset(cell for cell, scopes in self.membership.items() if scope_ref in scopes)

    def canonical_bytes(self) -> bytes:
        """Return a stable serialisation of this index as bytes.

        The output is byte-identical across instances and repeated calls
        for the same logical content (the byte-stable determinism gate for
        the scope fold).

        Format
        ------
        For each cell in ``sorted(self.membership)``:
        - the cell ref encoded as UTF-8
        - followed by ``:`` + each scope ref (sorted, UTF-8) in turn
        - followed by ``\\n``

        An empty membership produces ``b""``.

        Hash refs are fixed-width 64-char SHA-256 hex, so newline/colon
        framing is unambiguous.  Mirrors ``VectorIndex.canonical_bytes``.

        Returns
        -------
        bytes
        """
        out = bytearray()
        for cell in sorted(self.membership):
            out += cell.encode("utf-8")
            for scope in sorted(self.membership[cell]):
                out += b":" + scope.encode("utf-8")
            out += b"\n"
        return bytes(out)


# ---------------------------------------------------------------------------
# Pure fold function
# ---------------------------------------------------------------------------


def fold_scope(kernel: Any) -> ScopeIndex:
    """Fold ``kernel.all_events()`` into a direct-edge :class:`ScopeIndex`.

    Pure (D1/D4)
    ------------
    * Reads ``kernel.all_events()`` fresh on every call.
    * Persists no state between calls.
    * Appends no events to the kernel.

    Parameters
    ----------
    kernel:
        Any storage kernel that exposes
        ``all_events() -> list[tuple[SeqPos, Event]]``.

    Returns
    -------
    ScopeIndex
        ``membership`` contains one entry per ``cell_ref`` that appears as
        ``from_ref`` in at least one ``RelationshipEvent(type=SCOPED_TO)``.
        Values are ``frozenset`` of directly-attached scope refs (F3).
    """
    from amplifier_data.models import RelationshipEvent

    collected: dict[str, set[str]] = {}
    for _seq_pos, event in kernel.all_events():
        if isinstance(event, RelationshipEvent) and event.type == SCOPED_TO:
            collected.setdefault(event.from_ref, set()).add(event.to_ref)

    membership = {cell: frozenset(scopes) for cell, scopes in collected.items()}
    return ScopeIndex(membership=membership)
