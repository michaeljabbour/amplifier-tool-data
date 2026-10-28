"""GroupedLens — a grouped/inverted pure fold over the SAME events.

Design Decision D1 compliance
------------------------------
``project()`` folds ``kernel.all_events()`` on every call and persists **no**
mutated state between calls.  Two ``GroupedLens`` instances over the same
kernel are fully independent.

Grouped view
------------
GroupedLens selects the same cell-defining events as RelationalLens
(``CellWriteEvent`` and ``InterpreterOutputEvent``) and folds them into a
**grouped / inverted** shape that is structurally different from the flat rows
that RelationalLens produces.

The grouping key is ``event_type`` (``"cell_write"`` or
``"interpreter_output"``).  Each entry in the group list carries the same
fields as a RelationalLens row — including the **verbatim, unmodified
payload bytes** — which provides the E2 'different shape, same bytes' proof:
the payload bytes reachable through this view are identical to those reachable
through RelationalLens; only the structural organisation differs.

View type (lens-local)::

    View = dict[str, list[dict[str, Any]]]
    # e.g.
    # {
    #   "cell_write":         [{"seq_pos": 0, "payload": b"...", ...}, ...],
    #   "interpreter_output": [{"seq_pos": 2, "payload": b"...", ...}, ...],
    # }

query() typed surface
---------------------
``query(kernel=..., group=...)``

* ``kernel`` (required) — the storage kernel; ``project()`` is called fresh
  on each ``query()`` invocation.
* ``group`` (optional) — a group key string to restrict output to a single
  group.  ``None`` → all groups returned.  **An unknown group key raises
  ``KeyError`` immediately** — never a silent empty.

Returns a ToolResult-shaped ``QueryResult`` (``success``, ``output``,
``error``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from amplifier_data.lenses._scope import fold_scope
from amplifier_data.models import CellWriteEvent, InterpreterOutputEvent

# ---------------------------------------------------------------------------
# Known group keys
# ---------------------------------------------------------------------------

_KNOWN_GROUP_KEYS: frozenset[str] = frozenset({"cell_write", "interpreter_output"})

# ---------------------------------------------------------------------------
# Lens-local type aliases  (narrow the base aliases from amplifier_data.lenses)
# ---------------------------------------------------------------------------

View = dict[str, list[dict[str, Any]]]
"""Grouped table produced by :meth:`GroupedLens.project`.

Keys are ``event_type`` strings; values are lists of entry dicts (same field
set as RelationalLens rows) carrying **verbatim payload bytes**.
"""


@dataclass(frozen=True)
class QueryResult:
    """ToolResult-shaped result from :meth:`GroupedLens.query`.

    Mirrors the Amplifier ``ToolResult`` shape (reuse, not reinvent):
    ``success``, ``output``, ``error``.
    """

    success: bool
    output: dict[str, list[dict[str, Any]]]
    error: str | None = None


Result = QueryResult


# ---------------------------------------------------------------------------
# GroupedLens
# ---------------------------------------------------------------------------


class GroupedLens:
    """Grouped/inverted pure fold over the event log implementing the Lens Protocol.

    GroupedLens selects the same events as RelationalLens and presents the
    identical underlying payload bytes in a structurally different (grouped)
    shape, satisfying the E2 'different shape, same bytes' constraint.

    Example
    -------
    ::

        from amplifier_data.lenses.grouped import GroupedLens

        lens = GroupedLens()
        view = lens.project(kernel)
        # view == {
        #   "cell_write":         [{"seq_pos": 0, "payload": b"...", ...}, ...],
        #   "interpreter_output": [{"seq_pos": 2, "payload": b"...", ...}, ...],
        # }

        result = lens.query(kernel=kernel, group="cell_write")
        # result.success → True
        # result.output  → {"cell_write": [...]}
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this lens type."""
        return "grouped"

    def project(self, kernel: Any) -> View:
        """Fold ``kernel.all_events()`` into a grouped/inverted view.

        Pure — persists **no** mutated state between calls.  Each invocation
        re-folds the complete event log from scratch.

        Only ``CellWriteEvent`` and ``InterpreterOutputEvent`` become entries.
        ``RelationshipEvent`` and ``ToolResultEvent`` are not cell-defining
        and are silently skipped (same selection as RelationalLens).

        The grouping key is ``event_type``.  Payload bytes are verbatim and
        unmodified — the same bytes accessible through RelationalLens.

        Parameters
        ----------
        kernel:
            A storage kernel exposing
            ``all_events() -> list[tuple[SeqPos, Event]]``.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Freshly computed grouped view.  Mutating the returned dict does
            not affect subsequent ``project()`` calls.
        """
        view: View = {}
        for seq_pos, event in kernel.all_events():
            if isinstance(event, CellWriteEvent):
                entry: dict[str, Any] = {
                    "seq_pos": seq_pos,
                    "cell_ref": event.cell_ref(),
                    "payload": event.payload,
                    "event_type": "cell_write",
                    "interpreters": event.interpreters,
                    "model": None,
                    "input_hash": None,
                }
                view.setdefault("cell_write", []).append(entry)
            elif isinstance(event, InterpreterOutputEvent):
                entry = {
                    "seq_pos": seq_pos,
                    "cell_ref": event.cell_ref,
                    "payload": event.output,
                    "event_type": "interpreter_output",
                    "interpreters": (),
                    "model": event.model,
                    "input_hash": event.input_hash,
                }
                view.setdefault("interpreter_output", []).append(entry)
        return view

    def query(self, **typed_args: Any) -> QueryResult:
        """Typed surface over the grouped view.

        Keyword arguments
        -----------------
        kernel : Any  (required)
            The storage kernel to fold.  ``project()`` is called fresh on
            every ``query()`` invocation — no view is cached.
        group : str | None
            Optional group key to restrict output to a single group.
            ``None`` → all groups returned.  **An unknown group key raises
            ``KeyError`` immediately** — never a silent empty.

        Returns
        -------
        QueryResult
            ToolResult-shaped: ``success``, ``output`` (grouped dict),
            ``error``.

        Raises
        ------
        TypeError
            If ``kernel`` is not supplied.
        KeyError
            If ``group`` names a key not present in the known group keys.
        """
        if "kernel" not in typed_args:
            raise TypeError("GroupedLens.query() requires a 'kernel' keyword argument")

        kernel = typed_args["kernel"]
        group_key: str | None = typed_args.get("group")
        scope: str | None = typed_args.get("scope")

        # Validate group key before doing any work (fail fast, loud).
        if group_key is not None and group_key not in _KNOWN_GROUP_KEYS:
            raise KeyError(group_key)

        # Pure fold — always recomputes from the log; never reads cached state.
        view = self.project(kernel)

        # Scope filter (F3/F6): keep only entries whose cell_ref has a direct
        # SCOPED_TO edge to *scope*. An unknown scope yields empty output and
        # success=True — never raises. Mirrors RelationalLens.query.
        if scope is not None:
            scope_index = fold_scope(kernel)
            view = {
                gkey: [e for e in entries if scope_index.is_in_scope(e["cell_ref"], scope)]
                for gkey, entries in view.items()
            }
            view = {gkey: entries for gkey, entries in view.items() if entries}

        if group_key is not None:
            # If the group key is known but not present in this view (no events
            # of that type have been appended), return an empty group — the key
            # IS known; there are just no entries yet.  An *unknown* key was
            # already rejected above.
            output: View = {group_key: view.get(group_key, [])}
            if not output[group_key]:
                # Return empty output when there are no matching events.
                output = {}
        else:
            output = view

        return QueryResult(success=True, output=output)
