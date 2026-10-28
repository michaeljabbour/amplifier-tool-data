"""RelationalLens — a flat-rows pure fold over the event log.

Design Decision D1 compliance
------------------------------
``project()`` folds ``kernel.all_events()`` on every call and persists **no**
mutated state between calls.  Two ``RelationalLens`` instances over the same
kernel are fully independent.

Row schema
----------
Each cell-defining event (``CellWriteEvent`` or ``InterpreterOutputEvent``)
becomes one row with the following columns:

    seq_pos      : int           — SeqPos in the kernel log
    cell_ref     : Hash          — content-addressed identity
    payload      : bytes         — verbatim payload bytes (never reshaped)
    event_type   : str           — "cell_write" | "interpreter_output"
    interpreters : tuple[Hash,…] — interpreter refs (empty for interp events)
    model        : str | None    — model id (interpreter_output only)
    input_hash   : Hash | None   — input_hash (interpreter_output only)

``RelationshipEvent`` and ``ToolResultEvent`` are not cell-defining and are
excluded from the view.

query() typed surface
---------------------
``query(kernel=..., filter=..., project=...)``

* ``kernel`` (required) — the storage kernel; ``project()`` is called fresh
  on each ``query()`` invocation.
* ``filter`` (optional) — callable ``(row: dict) -> bool``; selects rows by
  any field.  ``None`` ⇒ all rows.
* ``project`` (optional) — list of column names to include in each output
  row.  ``None`` ⇒ all columns.  An unknown column name raises ``KeyError``
  immediately — never a silent empty.

Returns a ToolResult-shaped ``QueryResult`` (``success``, ``output``,
``error``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from amplifier_data.lenses._scope import fold_scope
from amplifier_data.models import CellWriteEvent, InterpreterOutputEvent

# ---------------------------------------------------------------------------
# Known row fields
# ---------------------------------------------------------------------------

_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "seq_pos",
        "cell_ref",
        "payload",
        "event_type",
        "interpreters",
        "model",
        "input_hash",
    }
)


# ---------------------------------------------------------------------------
# Lens-local type aliases  (narrow the base aliases from amplifier_data.lenses)
# ---------------------------------------------------------------------------

View = list[dict[str, Any]]
"""Flat table produced by :meth:`RelationalLens.project` — one dict per cell."""


@dataclass(frozen=True)
class QueryResult:
    """ToolResult-shaped result from :meth:`RelationalLens.query`.

    Mirrors the Amplifier ``ToolResult`` shape (reuse, not reinvent):
    ``success``, ``output``, ``error``.
    """

    success: bool
    output: list[dict[str, Any]]
    error: str | None = None


Result = QueryResult


# ---------------------------------------------------------------------------
# RelationalLens
# ---------------------------------------------------------------------------


class RelationalLens:
    """Flat-rows pure fold over the event log implementing the Lens Protocol.

    Example
    -------
    ::

        from amplifier_data.lenses.relational import RelationalLens

        lens = RelationalLens()
        rows = lens.project(kernel)                   # all rows, all columns
        result = lens.query(
            kernel=kernel,
            filter=lambda r: r["payload"].startswith(b"prefix"),
            project=["cell_ref", "payload"],
        )
        # result.success → True
        # result.output  → [{"cell_ref": ..., "payload": ...}, ...]
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this lens type."""
        return "relational"

    def project(self, kernel: Any) -> View:
        """Fold ``kernel.all_events()`` into flat rows.

        Pure — persists **no** mutated state between calls.  Each invocation
        re-folds the complete event log from scratch.

        Only ``CellWriteEvent`` and ``InterpreterOutputEvent`` become rows.
        ``RelationshipEvent`` and ``ToolResultEvent`` are not cell-defining
        and are silently skipped.

        Parameters
        ----------
        kernel:
            A storage kernel exposing
            ``all_events() -> list[tuple[SeqPos, Event]]``.

        Returns
        -------
        list[dict[str, Any]]
            Freshly computed flat table.  Mutating the returned list does not
            affect subsequent ``project()`` calls.
        """
        rows: View = []
        for seq_pos, event in kernel.all_events():
            if isinstance(event, CellWriteEvent):
                rows.append(
                    {
                        "seq_pos": seq_pos,
                        "cell_ref": event.cell_ref(),
                        "payload": event.payload,
                        "event_type": "cell_write",
                        "interpreters": event.interpreters,
                        "model": None,
                        "input_hash": None,
                    }
                )
            elif isinstance(event, InterpreterOutputEvent):
                rows.append(
                    {
                        "seq_pos": seq_pos,
                        "cell_ref": event.cell_ref,
                        "payload": event.output,
                        "event_type": "interpreter_output",
                        "interpreters": (),
                        "model": event.model,
                        "input_hash": event.input_hash,
                    }
                )
        return rows

    def query(self, **typed_args: Any) -> QueryResult:
        """Typed surface over the flat table.

        Keyword arguments
        -----------------
        kernel : Any  (required)
            The storage kernel to fold.  ``project()`` is called fresh on
            every ``query()`` invocation — no view is cached.
        filter : Callable[[dict[str, Any]], bool] | None
            Optional row predicate.  ``None`` ⇒ all rows included.
        project : Sequence[str] | None
            Column names to include in each output row.  ``None`` ⇒ all
            columns.  **An unknown column name raises ``KeyError``
            immediately** — never a silent empty.
        scope : Hash | None
            Optional scope reference.  When supplied, only rows whose
            ``cell_ref`` has a direct ``SCOPED_TO`` edge to *scope* are
            included (F3: direct edges only, no transitive closure).
            An unknown scope (no member cells) yields an empty output with
            ``success=True`` — never raises (F6).  Applied before *filter*
            and *project*.

        Returns
        -------
        QueryResult
            ToolResult-shaped: ``success``, ``output`` (list of row dicts),
            ``error``.

        Raises
        ------
        TypeError
            If ``kernel`` is not supplied.
        KeyError
            If any name in ``project`` is not a known row field.
        """
        if "kernel" not in typed_args:
            raise TypeError("RelationalLens.query() requires a 'kernel' keyword argument")

        kernel = typed_args["kernel"]
        filter_fn: Callable[[dict[str, Any]], bool] | None = typed_args.get("filter")
        project_fields: Sequence[str] | None = typed_args.get("project")
        scope: str | None = typed_args.get("scope")

        # Validate projection fields before doing any work (fail fast, loud).
        if project_fields is not None:
            for field in project_fields:
                if field not in _ROW_FIELDS:
                    raise KeyError(field)

        # Pure fold — always recomputes from the log; never reads cached state.
        rows = self.project(kernel)

        # Apply scope filter (F3/F6) — direct SCOPED_TO edges only.
        # Unknown scope yields empty output + success=True (F6: never raises).
        if scope is not None:
            scope_index = fold_scope(kernel)
            rows = [r for r in rows if scope_index.is_in_scope(r["cell_ref"], scope)]

        # Apply row filter.
        if filter_fn is not None:
            rows = [r for r in rows if filter_fn(r)]

        # Apply column projection.
        if project_fields is not None:
            rows = [{f: r[f] for f in project_fields} for r in rows]

        return QueryResult(success=True, output=rows)
