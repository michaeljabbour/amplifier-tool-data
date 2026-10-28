"""IndexLens — a static reflexive fold of AccessEvents into a regenerable IndexCell.

Design Decision D2 / D3 / D4 compliance
-----------------------------------------
``project()`` performs an EXPLICIT STATIC FOLD over the recorded AccessEvents
from ``kernel.all_events()`` and produces the index Cell's DEFINING EVENT
(an ``IndexCellEvent``, per D2).  The Cell is materialised via
``replay.regenerate()`` and is byte-identical across identical kernels —
a deterministic pure fold with **no mutable side-car**.

The fold is provenance-neutral: it reads the raw log via ``kernel.all_events()``
and never calls ``store.get_cell()``, satisfying the D4 read-vs-fold boundary —
folding the index produces **zero new AccessEvents**.

View shape
----------
``project()`` returns a ``Cell`` whose ``payload`` is a canonical JSON encoding
of the access count table (keys sorted for determinism)::

    {
        "<sha256-hex>": 3,   # this cell was read 3 times
        "<sha256-hex>": 1,   # this cell was read once
    }

The same bytes are carried in ``IndexCellEvent.access_summary`` — the
'why I indexed this' provenance.  The defining event is the sole entry
in ``cell.provenance``.

query() typed surface
---------------------
``query(kernel=..., cell_ref=...)``

* ``kernel`` (required) — the storage kernel; ``project()`` is called fresh on
  each ``query()`` invocation.
* ``cell_ref`` (optional) — restrict output to a single cell.  ``None`` → all
  cells returned.  If the cell has never been read its count is returned as
  ``0`` and it appears in the output only if explicitly requested.

Returns a ToolResult-shaped ``QueryResult`` (``success``, ``output``, ``error``).

Error handling
--------------
A ``Cell`` referenced by its ``cell_ref`` whose ``IndexCellEvent`` is absent
from the kernel log will cause ``replay.regenerate()`` to raise
``MissingDefiningEvent`` with the specific missing ``cell_ref`` — exactly as
Plan 1's regeneration guarantee already does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from amplifier_data.lenses._scope import fold_scope
from amplifier_data.models import AccessEvent, Cell, Hash, IndexCellEvent
from amplifier_data.replay import regenerate

# ---------------------------------------------------------------------------
# Lens-local type aliases
# ---------------------------------------------------------------------------

View = Cell
"""Materialised IndexCell produced by :meth:`IndexLens.project`.

The Cell's ``payload`` is a canonical JSON encoding of the access count table.
Its ``provenance`` contains the ``IndexCellEvent`` (the defining event) whose
``access_summary`` carries the same bytes as 'why I indexed this' provenance.
"""


@dataclass(frozen=True)
class QueryResult:
    """ToolResult-shaped result from :meth:`IndexLens.query`.

    Mirrors the Amplifier ``ToolResult`` shape (reuse, not reinvent):
    ``success``, ``output``, ``error``.
    """

    success: bool
    output: dict[Hash, int]
    error: str | None = None


Result = QueryResult


# ---------------------------------------------------------------------------
# IndexLens
# ---------------------------------------------------------------------------


class IndexLens:
    """Static reflexive fold of AccessEvents into a regenerable IndexCell.

    ``project()`` folds ``AccessEvent`` entries from ``kernel.all_events()``
    into an ``IndexCellEvent`` (the index Cell's defining event, per D2).
    The Cell is materialised via ``replay.regenerate()`` and is byte-identical
    across identical kernels — the fold is a pure deterministic function of the
    event log.

    The fold is provenance-neutral: reads the raw log via
    ``kernel.all_events()`` and never calls ``store.get_cell()``, satisfying
    the D4 read-vs-fold boundary (folding produces zero new AccessEvents).
    No mutable side-car is kept — the result is computed fresh on every call.

    Example
    -------
    ::

        from amplifier_data.lenses.index import IndexLens

        lens = IndexLens()
        cell = lens.project(kernel)
        # cell.payload          → canonical JSON access count table bytes
        # cell.provenance[0]    → IndexCellEvent (the defining event)
        # cell.provenance[0].access_summary  → same JSON bytes (provenance)

        result = lens.query(kernel=kernel, cell_ref="<hex>")
        # result.success → True
        # result.output  → {"<hex>": 3}
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this lens type."""
        return "index"

    def project(self, kernel: Any) -> Cell:
        """Fold ``kernel.all_events()`` into a regenerable IndexCell.

        Pure — persists **no** mutated state between calls and does NOT append
        any events to the kernel.  Each invocation re-folds the complete event
        log from scratch.

        Only ``AccessEvent`` entries contribute to the fold.  All other event
        types are silently skipped.

        The returned Cell:

        * ``payload``: canonical JSON encoding of ``{cell_ref: count}`` (keys
          sorted, compact separators — deterministic across Python versions)
        * ``provenance``: ``(IndexCellEvent,)`` — the sole defining event,
          whose ``access_summary`` carries the same bytes as the payload
          (the 'why I indexed this' provenance)

        Calling ``replay.regenerate([(0, cell.provenance[0])], cell_ref)``
        reproduces the byte-identical Cell.

        Parameters
        ----------
        kernel:
            A storage kernel exposing
            ``all_events() -> list[tuple[SeqPos, Event]]``.

        Returns
        -------
        Cell
            Freshly computed regenerable IndexCell.
        """
        # ── Step 1: fold AccessEvents into a count table ─────────────────────
        counts: dict[str, int] = {}
        for _seq_pos, event in kernel.all_events():
            if isinstance(event, AccessEvent):
                counts[event.cell_ref] = counts.get(event.cell_ref, 0) + 1

        # ── Step 2: canonical serialisation (sorted keys for determinism) ────
        summary_bytes: bytes = json.dumps(
            {k: counts[k] for k in sorted(counts)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        # ── Step 3: for this lens the access_summary IS the output payload ───
        #    Both carry the same canonical JSON count table.  A richer lens
        #    could separate provenance from the derived index structure; here
        #    they are identical bytes by design.
        output_bytes: bytes = summary_bytes

        # ── Step 4: derive content-addressed cell_ref (excludes cell_ref to
        #    avoid circularity — mirrors InterpreterOutputEvent.derive_ref) ───
        cell_ref: Hash = IndexCellEvent.derive_ref(
            access_summary=summary_bytes,
            output=output_bytes,
        )

        # ── Step 5: construct the defining event in memory (no kernel writes)─
        defining_event = IndexCellEvent(
            cell_ref=cell_ref,
            access_summary=summary_bytes,
            output=output_bytes,
        )

        # ── Step 6: materialise via regenerate() over a synthetic log ────────
        #    MissingDefiningEvent is impossible here — we just constructed the
        #    defining event.  The synthetic log avoids any kernel mutation.
        return regenerate([(0, defining_event)], cell_ref)

    def query(self, **typed_args: Any) -> QueryResult:
        """Typed surface over the access count table.

        Calls ``project()`` fresh on every invocation — no view is cached.

        Keyword arguments
        -----------------
        kernel : Any  (required)
            The storage kernel to fold.  ``project()`` is called fresh on
            every ``query()`` invocation — no view is cached.
        cell_ref : Hash | None
            Optional cell reference to restrict output to a single cell.
            ``None`` → all cells with at least one access are returned.
            If the cell has never been read and ``cell_ref`` is provided,
            ``{cell_ref: 0}`` is returned.

        Returns
        -------
        QueryResult
            ToolResult-shaped: ``success``, ``output`` (count dict),
            ``error``.

        Raises
        ------
        TypeError
            If ``kernel`` is not supplied.
        """
        if "kernel" not in typed_args:
            raise TypeError("IndexLens.query() requires a 'kernel' keyword argument")

        kernel = typed_args["kernel"]
        cell_ref: Hash | None = typed_args.get("cell_ref")
        scope: Hash | None = typed_args.get("scope")

        # Pure fold — always recomputes from the log; never reads cached state.
        index_cell: Cell = self.project(kernel)

        # Deserialise the Cell payload to recover the count table.
        counts: dict[Hash, int] = json.loads(index_cell.payload.decode("utf-8"))

        # Scope filter (F3/F6): restrict counts to cells with a direct SCOPED_TO
        # edge to *scope*. The reflexive fold itself is unchanged — scope is a
        # read-time filter on the output. Unknown scope yields {} (never raises).
        if scope is not None:
            scope_index = fold_scope(kernel)
            counts = {cr: n for cr, n in counts.items() if scope_index.is_in_scope(cr, scope)}

        if cell_ref is not None:
            output: dict[Hash, int] = {cell_ref: counts.get(cell_ref, 0)}
        else:
            output = counts

        return QueryResult(success=True, output=output)
